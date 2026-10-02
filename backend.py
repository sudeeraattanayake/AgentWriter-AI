from __future__ import annotations

import re

import base64

import os

from openai import OpenAI


import operator

from pathlib import Path

from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END

from langgraph.types import Send

from langchain_core.messages import SystemMessage, HumanMessage

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_openai import ChatOpenAI

import psycopg

from psycopg.rows import dict_row

from langgraph.checkpoint.postgres import PostgresSaver

from dotenv import load_dotenv

import os


load_dotenv()


def get_database_url():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:

        raise ValueError(



            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"



        )

    if "sslmode=" not in database_url:

        separator = "&" if "?" in database_url else "?"

        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not OPENAI_API_KEY:

    raise ValueError(



        "OPENAI_API_KEY is missing. Please add your OpenAI API key to .env"



    )


# ============================================================

# SANITIZATION

# ============================================================


def sanitize_null_chars(value):

    if isinstance(value, str):

        return value.replace("\x00", "")

    if isinstance(value, dict):

        return {

            key: sanitize_null_chars(val)

            for key, val in value.items()

        }

    if isinstance(value, list):

        return [

            sanitize_null_chars(item)

            for item in value

        ]

    if isinstance(value, tuple):

        return tuple(

            sanitize_null_chars(item)

            for item in value

        )

    return value


# Schema for the blog plan and tasks


class Task(BaseModel):

    id: int

    title: str

    goal: str = Field(..., description="One sentence describing what the reader should be able to do/understand after this section.",)

    bullets: List[str] = Field(..., min_length=3, max_length=6,

                               description="3–6 concrete, non-overlapping subpoints to cover in this section.",)

    target_words: int = Field(...,

                              description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)

    requires_research: bool = False

    requires_citations: bool = False

    requires_code: bool = False


class Plan(BaseModel):

    blog_title: str

    audience: str

    tone: str

    blog_kind: Literal["explainer", "tutorial", "news_roundup",



                       "comparison", "system_design"] = "explainer"

    constraints: List[str] = Field(default_factory=list)

    tasks: List[Task]


class EvidenceItem(BaseModel):

    title: str

    url: str

    # keep if Tavily provides; DO NOT rely on it

    published_at: Optional[str] = None

    snippet: Optional[str] = None

    source: Optional[str] = None


class RouterDecision(BaseModel):

    needs_research: bool

    mode: Literal["closed_book", "hybrid", "open_book"]

    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):

    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):

    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")

    filename: str = Field(...,



                          description="Save under images/, e.g. qkv_flow.png")

    alt: str

    caption: str

    prompt: str = Field(..., description="Prompt to send to the image model.")

    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"

    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):

    md_with_placeholders: str

    images: List[ImageSpec] = Field(default_factory=list)


# State schema for the blog agent


class State(TypedDict):

    topic: str

    # routing / research

    mode: str

    needs_research: bool

    queries: List[str]

    evidence: List[EvidenceItem]

    plan: Optional[Plan]

    # workers

    sections: Annotated[List[tuple[int, str]],



                        operator.add]  # (task_id, section_md)

    # reducer/image

    merged_md: str

    md_with_placeholders: str

    image_specs: List[dict]

    final: str


llm = ChatOpenAI(



    model="gpt-5-mini",



    api_key=OPENAI_API_KEY



)


# Router (decide upfront)


ROUTER_SYSTEM = """You are a routing module for a technical blog planner.



Decide whether web research is needed BEFORE planning.



Modes:



- closed_book (needs_research=false):



  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).



- hybrid (needs_research=true):



  Mostly evergreen but needs up-to-date examples/tools/models to be useful.



- open_book (needs_research=true):



  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.



If needs_research=true:



- Output 3–10 high-signal queries.



- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").



- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.



"""


def router_node(state: State) -> dict:

    topic = state["topic"]

    decider = llm.with_structured_output(RouterDecision)

    decision = decider.invoke(



        [



            SystemMessage(content=ROUTER_SYSTEM),



            HumanMessage(content=f"Topic: {topic}"),



        ]



    )

    return {



        "needs_research": decision.needs_research,



        "mode": decision.mode,



        "queries": decision.queries,



    }


def route_next(state: State) -> str:

    return "research" if state["needs_research"] else "orchestrator"


# ============================================================

# RESEARCH - TAVILY

# ============================================================


def _tavily_search(query: str, max_results: int = 5) -> List[dict]:

    tool = TavilySearchResults(max_results=max_results)

    response = tool.invoke({"query": query})

    # Tavily can return:

    # {"results": [...]}

    if isinstance(response, dict):

        results = response.get("results", [])

    # Some versions return:

    # [{...}, {...}]

    elif isinstance(response, list):

        results = response

    else:

        results = []

    normalized: List[dict] = []

    for r in results or []:

        # Prevent:

        # AttributeError: 'str' object has no attribute 'get'

        if not isinstance(r, dict):

            continue

        normalized.append(

            {

                "title": r.get("title") or "",

                "url": r.get("url") or "",

                "snippet": r.get("content") or r.get("snippet") or "",

                "published_at": (

                    r.get("published_date")

                    or r.get("published_at")

                ),

                "source": r.get("source"),

            }

        )

    return normalized


RESEARCH_SYSTEM = """

You are a research synthesizer for technical writing.



Given raw web search results, produce a deduplicated list

of EvidenceItem objects.



Rules:



- Only include items with a non-empty url.



- Prefer relevant + authoritative sources

  (company blogs, docs, reputable outlets).



- If a published date is explicitly present in the result

  payload, keep it as YYYY-MM-DD.



- If the published date is missing or unclear,

  set published_at=null. Do NOT guess.



- Keep snippets short.



- Deduplicate by URL.

"""


def research_node(state: State) -> dict:

    # Take queries generated by the router

    queries = state.get("queries", []) or []

    max_results = 6

    raw_results: List[dict] = []

    for q in queries:

        results = _tavily_search(

            q,

            max_results=max_results,

        )

        raw_results.extend(results)

    # No research results

    if not raw_results:

        return {

            "evidence": []

        }

    # Convert raw Tavily results into structured evidence

    extractor = llm.with_structured_output(

        EvidencePack

    )

    pack = extractor.invoke(

        [

            SystemMessage(

                content=RESEARCH_SYSTEM

            ),

            HumanMessage(

                content=f"Raw results:\n{raw_results}"

            ),

        ]

    )

    # Deduplicate by URL

    dedup = {}

    for e in pack.evidence:

        if e.url:

            dedup[e.url] = e

    return {

        "evidence": list(dedup.values())

    }


# Orchestrator (Plan)


ORCH_SYSTEM = """You are a senior technical writer and developer advocate.



Your job is to produce a highly actionable outline for a technical blog post.



Hard requirements:



- Create 5–9 sections (tasks) suitable for the topic and audience.



- Each task must include:



  1) goal (1 sentence)



  2) 3–6 bullets that are concrete, specific, and non-overlapping



  3) target word count (120–550)



Quality bar:



- Assume the reader is a developer; use correct terminology.



- Bullets must be actionable: build/compare/measure/verify/debug.



- Ensure the overall plan includes at least 2 of these somewhere:



  - minimal code sketch / MWE (set requires_code=True for that section)



  - edge cases / failure modes



  - performance/cost considerations



  - security/privacy considerations (if relevant)



  - debugging/observability tips



Grounding rules:



- Mode closed_book: keep it evergreen; do not depend on evidence.



- Mode hybrid:



  - Use evidence for up-to-date examples (models/tools/releases) in bullets.



  - Mark sections using fresh info as requires_research=True and requires_citations=True.



- Mode open_book:



  - Set blog_kind = "news_roundup".



  - Every section is about summarizing events + implications.



  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.



  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"



    and includes only what can be supported.

Output must strictly match the Plan schema.



"""


def orchestrator_node(state: State) -> dict:

    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])

    mode = state.get("mode", "closed_book")

    plan = planner.invoke(



        [



            SystemMessage(content=ORCH_SYSTEM),



            HumanMessage(



                content=(



                    f"Topic: {state['topic']}\n"



                    f"Mode: {mode}\n\n"



                    f"Evidence (ONLY use for fresh claims; may be empty):\n"



                    f"{[e.model_dump() for e in evidence][:16]}"



                )



            ),



        ]



    )

    return {"plan": plan}


# Fanout


def fanout(state: State):

    return [



        Send(



            "worker",



            {



                "task": task.model_dump(),



                "topic": state["topic"],



                "mode": state["mode"],



                "plan": state["plan"].model_dump(),



                "evidence": [e.model_dump() for e in state.get("evidence", [])],



            },



        )



        for task in state["plan"].tasks



    ]


# Worker (write one section)


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.



Write ONE section of a technical blog post in Markdown.



Hard constraints:



- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).



- Stay close to Target words (±15%).



- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).



- Start with a '## <Section Title>' heading.



Scope guard:



- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.



  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.



  Focus on summarizing events and implications.



Grounding policy:



- If mode == open_book:



  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.



  - For each event claim, attach a source as a Markdown link: ([Source](URL)).



  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."



- If requires_citations == true:



  - For outside-world claims, cite Evidence URLs the same way.



- Evergreen reasoning is OK without citations unless requires_citations is true.



Code:



- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.



Style:



- Short paragraphs, bullets where helpful, code fences for code.



- Avoid fluff/marketing. Be precise and implementation-oriented.



"""


def worker_node(payload: dict) -> dict:

    task = Task(**payload["task"])

    plan = Plan(**payload["plan"])

    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    topic = payload["topic"]

    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""

    if evidence:

        evidence_text = "\n".join(



            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()



            for e in evidence[:20]



        )

    section_md = llm.invoke(



        [



            SystemMessage(content=WORKER_SYSTEM),



            HumanMessage(



                content=(



                    f"Blog title: {plan.blog_title}\n"



                    f"Audience: {plan.audience}\n"



                    f"Tone: {plan.tone}\n"



                    f"Blog kind: {plan.blog_kind}\n"



                    f"Constraints: {plan.constraints}\n"



                    f"Topic: {topic}\n"



                    f"Mode: {mode}\n\n"



                    f"Section title: {task.title}\n"



                    f"Goal: {task.goal}\n"



                    f"Target words: {task.target_words}\n"



                    f"Tags: {task.tags}\n"



                    f"requires_research: {task.requires_research}\n"



                    f"requires_citations: {task.requires_citations}\n"



                    f"requires_code: {task.requires_code}\n"



                    f"Bullets:{bullets_text}\n\n"



                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"



                )



            ),



        ]



    ).content.strip()

    section_md = sanitize_null_chars(section_md)

    return {"sections": [(task.id, section_md)]}


# ============================================================


# 8) ReducerWithImages (subgraph)


#    merge_content -> decide_images -> generate_and_place_images


# ============================================================


def merge_content(state: State) -> dict:

    plan = state["plan"]

    ordered_sections = [



        md for _, md in sorted(



            state["sections"],



            key=lambda x: x[0]



        )



    ]

    body = "\n\n".join(ordered_sections).strip()

    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    merged_md = sanitize_null_chars(merged_md)

    return {



        "merged_md": merged_md



    }


DECIDE_IMAGES_SYSTEM = """



You are an expert technical editor.

Decide if images/diagrams are needed for THIS blog.

Rules:



- Max 3 images total.



- Each image must materially improve understanding.



- Prefer diagrams, architecture visuals, flows, or educational illustrations.



- Insert placeholders exactly:



  [[IMAGE_1]]



  [[IMAGE_2]]



  [[IMAGE_3]]



- If no images are needed:



  md_with_placeholders must equal input and images=[].



- Avoid decorative images.



- Prefer technical diagrams with short, readable labels.



- filename must contain only the filename, for example:



  attention_pipeline.png



- Never include "images/" or any directory path in filename.



Return strictly GlobalImagePlan.



"""


def decide_images(state: State) -> dict:

    planner = llm.with_structured_output(GlobalImagePlan)

    merged_md = state["merged_md"]

    plan = state["plan"]

    assert plan is not None

    image_plan = planner.invoke(

        [



            SystemMessage(



                content=DECIDE_IMAGES_SYSTEM



            ),



            HumanMessage(



                content=(



                    f"Blog kind: {plan.blog_kind}\n"



                    f"Topic: {state['topic']}\n\n"



                    "Insert placeholders and propose image prompts.\n\n"



                    f"{merged_md}"



                )



            ),



        ]



    )

    return sanitize_null_chars({
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [
            img.model_dump()
            for img in image_plan.images
        ],
    })


# ============================================================


# OpenAI Image Generation


# ============================================================


def _openai_generate_image_bytes(prompt: str) -> bytes:
    """

    Generate an image using OpenAI GPT Image 2.5 Flare

    and return the raw PNG bytes.

    Requires:



        pip install openai

    Environment variable:



        OPENAI_API_KEY



    """

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:

        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(



        api_key=api_key



    )

    response = client.images.generate(

        model="gpt-image-2.5-flare",

        prompt=prompt,

        size="1536x1024",

        quality="high",

        output_format="png"

    )

    if not response.data:

        raise RuntimeError(

            "OpenAI returned no image data."

        )

    image_base64 = response.data[0].b64_json

    if not image_base64:

        raise RuntimeError(

            "OpenAI returned no base64 image data."

        )

    return base64.b64decode(image_base64)


# ============================================================


# Generate Images + Insert Into Markdown


# ============================================================


def generate_and_place_images(state: State) -> dict:

    plan = state["plan"]

    assert plan is not None

    md = (

        state.get("md_with_placeholders")

        or state["merged_md"]

    )

    image_specs = (

        state.get("image_specs", [])

        or []

    )

    # --------------------------------------------------------

    # Safe blog filename

    # --------------------------------------------------------

    safe_title = re.sub(

        r"[^a-zA-Z0-9\s-]",

        "",

        plan.blog_title

    )

    safe_title = safe_title.lower().strip()

    safe_title = re.sub(r"\s+", "_", safe_title)

    blog_filename = f"{safe_title}.md"

    # --------------------------------------------------------

    # No images requested

    # --------------------------------------------------------

    if not image_specs:

        md = sanitize_null_chars(md)

        Path(blog_filename).write_text(

            md,

            encoding="utf-8"

        )

        return {

            "final": md

        }

    # --------------------------------------------------------

    # Image directory

    # --------------------------------------------------------

    images_dir = Path("images")

    images_dir.mkdir(



        parents=True,



        exist_ok=True



    )

    # --------------------------------------------------------

    # Generate images

    # --------------------------------------------------------

    for spec in image_specs:

        placeholder = spec["placeholder"]

        # Keep only the actual filename.

        # Example:

        # images/attention_pipeline.png

        # becomes:

        # attention_pipeline.png

        filename = Path(spec["filename"]).name

        # Make sure filename has PNG extension

        if not filename.lower().endswith(".png"):

            filename = f"{filename}.png"

        out_path = images_dir / filename

        # Generate only when image doesn't already exist

        if not out_path.exists():

            try:

                img_bytes = _openai_generate_image_bytes(



                    spec["prompt"]



                )

                out_path.write_bytes(



                    img_bytes



                )

            except Exception as e:

                prompt_block = (



                    f"> **IMAGE GENERATION FAILED** "



                    f"{spec.get('caption', '')}\n>\n"



                    f"> **Alt:** "



                    f"{spec.get('alt', '')}\n>\n"



                    f"> **Prompt:** "



                    f"{spec.get('prompt', '')}\n>\n"



                    f"> **Error:** {e}\n"



                )

                md = md.replace(



                    placeholder,



                    prompt_block



                )

                continue

        # ----------------------------------------------------

        # Insert image into Markdown

        # ----------------------------------------------------

        img_md = (



            f"![{spec['alt']}](images/{filename})\n\n"



            f"*{spec['caption']}*"



        )

        md = md.replace(



            placeholder,



            img_md



        )

    # --------------------------------------------------------

    # Save final blog

    # --------------------------------------------------------

    md = sanitize_null_chars(md)

    Path(blog_filename).write_text(



        md,



        encoding="utf-8"



    )

    return {



        "final": md



    }


# Build the subgraph for merging content and generating images


reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_content", merge_content)

reducer_graph.add_node("decide_images", decide_images)

reducer_graph.add_node("generate_and_place_images", generate_and_place_images)

reducer_graph.add_edge(START, "merge_content")

reducer_graph.add_edge("merge_content", "decide_images")

reducer_graph.add_edge("decide_images", "generate_and_place_images")

reducer_graph.add_edge("generate_and_place_images", END)

reducer_subgraph = reducer_graph.compile()


reducer_subgraph


# Build the main graph

g = StateGraph(State)

g.add_node("router", router_node)

g.add_node("research", research_node)

g.add_node("orchestrator", orchestrator_node)

g.add_node("worker", worker_node)

g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")

g.add_conditional_edges("router", route_next, {

                        "research": "research", "orchestrator": "orchestrator"})

g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])

g.add_edge("worker", "reducer")

g.add_edge("reducer", END)


# =========================


# PostgreSQL Checkpointer


# =========================


DATABASE_URL = get_database_url()

_conn = psycopg.connect(

    DATABASE_URL,

    autocommit=True,

    row_factory=dict_row



)


checkpointer = PostgresSaver(_conn)


checkpointer.setup()


app = g.compile(checkpointer=checkpointer)


app
