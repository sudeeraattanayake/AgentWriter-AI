# ✍️ AgentWriter AI — Real-Time Multi-Agent Technical Content Generation Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/LangGraph-Multi--Agent_Workflow-1C3C3C" alt="LangGraph">
  <img src="https://img.shields.io/badge/OpenAI-LLM_%2B_Images-412991?logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/Tavily-Web_Research-FF6B35" alt="Tavily">
  <img src="https://img.shields.io/badge/FastAPI-Web_App-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/PostgreSQL-Checkpointing-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/SSE-Real--Time_Streaming-E34F26" alt="SSE">
  <img src="https://img.shields.io/badge/AI-Parallel_Agents-blueviolet" alt="Parallel Agents">
  <img src="https://img.shields.io/badge/UI-Red_Neon-FF1744" alt="Red Neon UI">
  <img src="https://img.shields.io/badge/Git-Version_Control-F05032?logo=git&logoColor=white" alt="Git">
  <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub">
</p>

<p align="center">
  <strong>
    An end-to-end multi-agent technical writing platform that routes requests, performs optional web research,
    creates structured article plans, writes sections in parallel, merges content, generates technical visuals,
    and streams execution progress through a LangGraph-powered workflow.
  </strong>
</p>

<p align="center">
  📝 <strong>Topic</strong> →
  🧭 <strong>Router</strong> →
  🔎 <strong>Research</strong> →
  🧠 <strong>Planner</strong> →
  ✍️ <strong>Parallel Writers</strong> →
  🔀 <strong>Reducer</strong> →
  🖼️ <strong>Visual Generator</strong> →
  📄 <strong>Final Article</strong>
</p>

<p align="center">
  <a href="#user-interface">Screenshots</a> ·
  <a href="#example-generated-article">Generated Blog</a> ·
  <a href="#installation">Installation</a>
</p>

---

## 📌 Overview

**AgentWriter AI** is an end-to-end multi-agent technical content generation application built using **Python, LangGraph, LangChain, OpenAI, Tavily, PostgreSQL, FastAPI, HTML, CSS, and JavaScript**.

Users provide a natural-language writing request such as:

> Write a practical technical article explaining LangGraph subgraphs, including architecture, code examples, failure modes, and production considerations.

AgentWriter AI processes the request through a specialized graph workflow:

1. **Router Agent** determines whether the topic requires external research and selects `closed_book`, `hybrid`, or `open_book` mode.
2. **Research Agent** uses Tavily when fresh or externally grounded information is required.
3. **Orchestrator / Planner** creates a structured article plan containing section goals, bullets, target word counts, tags, and research/code requirements.
4. **Writer Workers** generate planned sections independently and can execute through LangGraph fan-out.
5. **Reducer** sorts and merges the generated sections into a complete Markdown article.
6. **Image Planner** decides whether technical visuals materially improve the article.
7. **OpenAI Image Generation** creates selected diagrams or educational visuals and inserts them into the final Markdown.
8. **FastAPI + SSE** streams workflow activity to the browser while the graph executes.

The application uses **LangGraph `StateGraph`**, conditional routing, `Send`-based fan-out, a reducer subgraph, and a **PostgreSQL `PostgresSaver` checkpointer**.

A custom **red-neon HTML, CSS, and JavaScript interface** provides workflow activity, article strategy, Markdown rendering, syntax-highlighted code, copy/download controls, progress indicators, and generated-article previews.

---

## ✨ Features

- ✍️ AI-powered technical article generation.
- 🤖 Multi-agent workflow built with LangGraph.
- 🧭 Intelligent research routing.
- 📚 `closed_book`, `hybrid`, and `open_book` generation modes.
- 🔎 Tavily-powered web research.
- 🔗 Evidence-aware technical writing and citations.
- 🧠 Structured article planning with Pydantic models.
- 📋 Section goals, bullets, tags, target word counts, and constraints.
- ⚡ LangGraph fan-out with parallel writer workers.
- 🧩 Independent section generation.
- 💻 Code-aware section planning and generation.
- 🔀 Ordered reducer for combining parallel section results.
- 🖼️ AI image-planning stage.
- 🎨 OpenAI-powered image generation.
- 📝 Complete Markdown article output.
- 📥 Downloadable generated `.md` articles.
- 🗃️ PostgreSQL-backed LangGraph checkpointing.
- 📡 Server-Sent Events for real-time workflow updates.
- 🌐 FastAPI web application.
- ❤️ Application health-check endpoint.
- 🎨 Custom red-neon interface.
- ✨ 3D-style UI depth and glow effects.
- ⏳ Real-time agent activity timeline.
- 🧠 Live article-plan visualization.
- 📊 Workflow progress indicator.
- 📝 Markdown-rendered article preview.
- 🌈 Syntax-highlighted code blocks.
- 📋 Copy generated content.
- 📱 Responsive interface.
- 🔐 Environment-variable configuration.
- 🛡️ Null-character sanitization before persistent graph state.
- 🧩 Modular backend, API, frontend, notebooks, and assets structure.

---

<a id="user-interface"></a>

## 📸 User Interface

### AgentWriter AI Home

The AgentWriter AI home interface provides a focused red-neon technical writing workspace with article-topic input, workflow stages, server status, execution progress, strategy panels, and generated-content output.

<p align="center">
  <img src="assets/screenshots/home-1.png" alt="AgentWriter AI home interface" width="1000">
</p>

<p align="center">
  <img src="assets/screenshots/home-2.png" alt="AgentWriter AI workspace" width="1000">
</p>

### ⚡ Execution — Agent Activity

The execution workspace exposes the progression of the LangGraph workflow so users can follow routing, research, planning, section writing, reduction, and visual generation while the article is being produced.

<p align="center">
  <img src="assets/screenshots/agent-activity.png" alt="AgentWriter AI real-time agent activity" width="1000">
</p>

### 🧠 Strategy — Article Plan

The strategy panel presents the structured plan produced by the orchestrator, including the planned sections and writing requirements used by the worker agents.

<p align="center">
  <img src="assets/screenshots/article-plan.png" alt="AgentWriter AI article plan" width="1000">
</p>

### 📝 Generated Article

AgentWriter AI merges independently generated sections into a complete technical article and renders the Markdown directly inside the application.

<p align="center">
  <img src="assets/screenshots/generated-article-1.png" alt="AgentWriter AI generated article 1" width="1000">
</p>

<p align="center">
  <img src="assets/screenshots/generated-article-2.png" alt="AgentWriter AI generated article 2" width="1000">
</p>

<p align="center">
  <img src="assets/screenshots/generated-article-3.png" alt="AgentWriter AI generated article 3" width="1000">
</p>

<a id="example-generated-article"></a>

### 📖 Example Generated Blog

A complete Markdown article produced by AgentWriter AI is included in the repository as a real project output.

**[📄 Read the complete generated article](assets/examples/example.md)**

This example demonstrates the final readable Markdown output rather than only a screenshot of the application.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    USER["📝 User Topic"] --> UI["🎨 AgentWriter AI Web Interface"]
    UI --> API["🌐 FastAPI Application"]
    API --> GRAPH["🦜 LangGraph Workflow"]

    GRAPH --> ROUTER["🧭 Router Agent"]
    ROUTER -->|Research required| RESEARCH["🔎 Research Agent"]
    ROUTER -->|No research required| PLAN["🧠 Orchestrator / Planner"]

    RESEARCH --> TAVILY["🌐 Tavily Search"]
    TAVILY --> PLAN

    PLAN --> FANOUT["⚡ LangGraph Send Fan-Out"]

    FANOUT --> W1["✍️ Writer Worker 1"]
    FANOUT --> W2["✍️ Writer Worker 2"]
    FANOUT --> WN["✍️ Writer Worker N"]

    W1 --> REDUCER["🔀 Reducer Subgraph"]
    W2 --> REDUCER
    WN --> REDUCER

    REDUCER --> MERGE["📝 Merge Content"]
    MERGE --> IMAGEPLAN["🧠 Decide Images"]
    IMAGEPLAN --> IMAGEGEN["🖼️ OpenAI Image Generation"]
    IMAGEGEN --> FINAL["📄 Final Markdown Article"]

    FINAL --> API
    API -->|SSE Events| UI

    GRAPH -. Checkpoints .-> POSTGRES["🐘 PostgreSQL"]
```

The high-level workflow is:

```text
START
  ↓
🧭 Router
  ↓
Research needed?
  ├── Yes → 🔎 Tavily Research ──┐
  └── No ────────────────────────┤
                                 ↓
                         🧠 Orchestrator
                                 ↓
                         ⚡ Send / Fan-Out
                     ┌───────────┼───────────┐
                     ↓           ↓           ↓
                  ✍️ Worker   ✍️ Worker   ✍️ Worker
                     └───────────┼───────────┘
                                 ↓
                         🔀 Reducer Subgraph
                                 ↓
                           📝 Merge Content
                                 ↓
                           🖼️ Decide Images
                                 ↓
                      🎨 Generate / Place Images
                                 ↓
                         📄 Final Markdown
                                 ↓
                                END
```

---

## 🔄 How It Works

### 1️⃣ User Enters a Technical Writing Request

The user enters a topic through the AgentWriter AI interface.

Example:

> Explain LangGraph subgraphs with a practical multi-agent example, code, architecture, debugging guidance, and production considerations.

The frontend sends the request to the FastAPI application and opens an SSE stream for execution updates.

---

### 2️⃣ FastAPI Starts a Workflow Run

FastAPI creates a run identifier and invokes the compiled LangGraph workflow.

The graph receives an initial state containing fields for routing, evidence, planning, generated sections, merged Markdown, image specifications, and the final article.

---

### 3️⃣ Router Determines the Research Strategy

The first graph node is:

```python
router_node
```

The Router produces a structured `RouterDecision`:

```python
class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal[
        "closed_book",
        "hybrid",
        "open_book"
    ]
    queries: List[str] = Field(default_factory=list)
```

The routing modes are:

| Mode | Purpose |
| --- | --- |
| `closed_book` | Evergreen technical topics that do not require fresh external facts |
| `hybrid` | Mostly evergreen topics that benefit from current examples, tools, or releases |
| `open_book` | Fresh or volatile topics that require research-backed claims |

LangGraph then conditionally routes to either `research` or `orchestrator`.

---

### 4️⃣ Research Agent Retrieves Evidence

When research is required, AgentWriter AI executes Tavily searches generated by the Router.

Raw search results are normalized and then transformed into structured evidence:

```python
class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None
```

The research stage prefers relevant sources, keeps explicit publication dates when available, avoids guessing missing dates, and deduplicates evidence by URL.

The resulting evidence is passed to the Orchestrator.

---

### 5️⃣ Orchestrator Builds the Article Strategy

The Orchestrator uses structured output to produce a `Plan`.

```python
class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]
```

Each task contains:

```python
class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: List[str]
    target_words: int
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False
```

The plan becomes the strategy shown in the frontend.

---

### 6️⃣ LangGraph Fans Out Writer Tasks

AgentWriter AI uses `Send` to dispatch each planned section to the writer node.

Conceptually:

```python
return [
    Send(
        "worker",
        {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "plan": state["plan"].model_dump(),
            "evidence": [
                e.model_dump()
                for e in state.get("evidence", [])
            ],
        },
    )
    for task in state["plan"].tasks
]
```

This allows the graph to process independently planned article sections through the worker stage.

---

### 7️⃣ Writer Workers Generate Sections

Each worker receives one task and writes one Markdown section.

The worker follows:

- Section title.
- Section goal.
- Planned bullets.
- Target word count.
- Article audience and tone.
- Research requirements.
- Citation requirements.
- Code requirements.
- Available evidence.

Generated sections are returned as:

```python
{
    "sections": [
        (task.id, section_md)
    ]
}
```

The `sections` field uses a LangGraph reducer based on `operator.add`.

---

### 8️⃣ Reducer Orders and Merges Sections

Because worker results may complete independently, the reducer sorts sections by their task ID:

```python
ordered_sections = [
    md
    for _, md in sorted(
        state["sections"],
        key=lambda x: x[0]
    )
]
```

The article body is joined and combined with the planned blog title.

The result is stored in:

```text
merged_md
```

---

### 9️⃣ Image Planner Decides Whether Visuals Are Needed

The reducer subgraph passes the merged article to an image-planning stage.

The image planner can request up to three useful visuals and returns:

```python
class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str
    size: Literal[
        "1024x1024",
        "1024x1536",
        "1536x1024"
    ] = "1024x1024"
    quality: Literal[
        "low",
        "medium",
        "high"
    ] = "medium"
```

The planner favors architecture diagrams, flows, and educational visuals rather than decorative images.

---

### 🔟 OpenAI Generates and Places Images

When image specifications exist, AgentWriter AI generates PNG images through the OpenAI Images API.

Each placeholder such as:

```text
[[IMAGE_1]]
```

is replaced with Markdown referencing the generated visual:

```markdown
![Architecture diagram](images/example.png)

*Architecture diagram caption*
```

If no images are needed, the merged Markdown continues unchanged.

---

### 1️⃣1️⃣ Final Markdown Is Saved

AgentWriter AI creates a safe filename from the generated blog title and saves the final article as UTF-8 Markdown.

The application can then present and download the generated `.md` output.

---

### 1️⃣2️⃣ PostgreSQL Persists LangGraph Checkpoints

The graph is compiled with `PostgresSaver`:

```python
checkpointer = PostgresSaver(connection)
checkpointer.setup()

app = graph.compile(
    checkpointer=checkpointer
)
```

Checkpointing allows workflow state to be associated with LangGraph thread configuration.

Generated strings are sanitized for null characters before relevant persistent state is returned, preventing PostgreSQL text/JSON failures caused by `\u0000`.

---

### 1️⃣3️⃣ FastAPI Streams Progress to the Browser

The web application emits Server-Sent Events for important workflow stages.

The interface can update the execution timeline as the graph progresses through:

```text
Router
Research
Plan
Workers
Reducer
Image Planning
Final Output
```

This gives the user visibility into the multi-agent process instead of waiting for a single final response.

---

## 🧠 Agents and Responsibilities

| Component | Type | Responsibility |
| --- | --- | --- |
| 🧭 Router | Routing node | Decide whether research is required and select generation mode |
| 🔎 Research Agent | Research node | Retrieve and normalize Tavily evidence |
| 🧠 Orchestrator | Planning node | Create the structured article plan |
| ✍️ Writer Worker | Generation node | Write one planned Markdown section |
| ⚡ Fan-Out | LangGraph routing | Dispatch planned tasks through `Send` |
| 🔀 Reducer | Subgraph | Merge sections and coordinate image stages |
| 📝 Merge Content | Reducer node | Sort worker outputs and create the article body |
| 🖼️ Image Planner | Structured LLM node | Decide whether useful technical visuals are needed |
| 🎨 Image Generator | Image node | Generate and place OpenAI images |
| 🐘 PostgreSQL Checkpointer | Persistence | Store LangGraph checkpoints |
| 🌐 FastAPI | Application layer | Serve the UI and workflow API |
| 📡 SSE | Streaming layer | Send real-time execution events to the browser |

---

## 🧠 Shared Agent State

AgentWriter AI uses a typed state shared across the graph:

```python
class State(TypedDict):
    topic: str

    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]

    plan: Optional[Plan]

    sections: Annotated[
        List[tuple[int, str]],
        operator.add
    ]

    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str
```

| State Field | Purpose |
| --- | --- |
| `topic` | Original technical writing request |
| `mode` | Router-selected research mode |
| `needs_research` | Determines whether the research node executes |
| `queries` | Web-search queries generated by the Router |
| `evidence` | Structured research evidence |
| `plan` | Structured article strategy |
| `sections` | Parallel worker outputs |
| `merged_md` | Ordered article before image placement |
| `md_with_placeholders` | Article containing image placeholders |
| `image_specs` | Structured image-generation instructions |
| `final` | Final Markdown article |

---

## 🦜 LangGraph Workflow Composition

The main graph is constructed using `StateGraph`:

```python
g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")

g.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator",
    },
)

g.add_edge("research", "orchestrator")
g.add_conditional_edges(
    "orchestrator",
    fanout,
    ["worker"]
)
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)
```

The reducer is itself a LangGraph subgraph:

```python
reducer_graph = StateGraph(State)

reducer_graph.add_node(
    "merge_content",
    merge_content
)

reducer_graph.add_node(
    "decide_images",
    decide_images
)

reducer_graph.add_node(
    "generate_and_place_images",
    generate_and_place_images
)

reducer_graph.add_edge(
    START,
    "merge_content"
)

reducer_graph.add_edge(
    "merge_content",
    "decide_images"
)

reducer_graph.add_edge(
    "decide_images",
    "generate_and_place_images"
)

reducer_graph.add_edge(
    "generate_and_place_images",
    END
)
```

This separates article reduction and visual generation from the main orchestration graph.

---

## 🔎 Tavily Research

AgentWriter AI uses Tavily when the Router determines that a topic needs external grounding.

The research layer:

- Executes multiple high-signal search queries.
- Normalizes different Tavily response shapes.
- Filters malformed results.
- Preserves URLs.
- Keeps explicit publication dates where available.
- Avoids inventing missing dates.
- Deduplicates evidence by URL.
- Supplies evidence to later planning and writing stages.

Research can therefore be skipped for evergreen topics while remaining available for fresh or volatile subjects.

---

## 🤖 OpenAI Integration

AgentWriter AI uses OpenAI through `langchain-openai` for structured reasoning and technical content generation.

The core language model is configured conceptually as:

```python
llm = ChatOpenAI(
    model="gpt-5-mini",
    api_key=OPENAI_API_KEY
)
```

Structured output is used for:

- Routing decisions.
- Evidence synthesis.
- Article plans.
- Image plans.

Normal model invocation is used for individual article sections.

The image-generation stage uses the OpenAI Images API and requires access to the image model configured in `backend.py`.

---

## 🐘 PostgreSQL Checkpointing

AgentWriter AI uses PostgreSQL for LangGraph checkpoint persistence.

The application reads:

```text
DATABASE_URL
```

from environment configuration and ensures SSL mode is present when required.

The checkpointer is initialized using:

```python
PostgresSaver
```

This persistence layer stores LangGraph checkpoint state rather than acting as the article-content database.

---

## 📡 Real-Time SSE Streaming

Instead of waiting silently for the complete graph to finish, AgentWriter AI exposes workflow progress through **Server-Sent Events (SSE)**.

The frontend can display:

- Current graph stage.
- Routing result.
- Research progress.
- Article plan.
- Completed writer sections.
- Reducer activity.
- Image planning.
- Final article completion.

This makes the graph execution visible and easier to understand during long-running article generation.

---

## 🛠️ Technologies Used

| Technology | Purpose |
| --- | --- |
| 🐍 Python | Core application language |
| 🦜 LangChain | OpenAI integration and messages |
| 🔀 LangGraph | Multi-agent graph orchestration |
| 🤖 OpenAI | Technical writing and structured generation |
| 🎨 OpenAI Images | AI-generated article visuals |
| 🔎 Tavily | External web research |
| 🐘 PostgreSQL | LangGraph checkpoint persistence |
| 💾 PostgresSaver | LangGraph PostgreSQL checkpointer |
| 🌐 FastAPI | Backend API and web application |
| ⚡ Uvicorn | ASGI application server |
| 📡 SSE | Real-time workflow event streaming |
| 📄 Jinja2 | Frontend template rendering |
| 🧱 Pydantic | Structured agent outputs |
| 🔐 python-dotenv | Environment-variable loading |
| 🎨 HTML + CSS | Interface structure and red-neon design |
| ⚙️ JavaScript | Streaming, UI state, and interactions |
| 📝 Marked | Markdown article rendering |
| 🛡️ DOMPurify | Sanitization of rendered Markdown HTML |
| 🌈 Highlight.js | Syntax highlighting for generated code |
| 🧰 Git | Version control |
| 🐙 GitHub | Source-code hosting |

---

## 📦 Main Python Libraries

```text
fastapi
uvicorn
jinja2
pydantic
python-dotenv
langchain
langchain-core
langchain-openai
langchain-community
langgraph
langgraph-checkpoint-postgres
psycopg
openai
tavily-python
```

Install the exact versions required by the project using `requirements.txt`.

---

## 📁 Project Structure

```text
AgentWriter-AI/
│
├── assets/
│   ├── examples/
│   │   └── example.md
│   │
│   └── screenshots/
│       ├── agent-activity.png
│       ├── article-plan.png
│       ├── generated-article-1.png
│       ├── generated-article-2.png
│       ├── generated-article-3.png
│       ├── home-1.png
│       └── home-2.png
│
├── notebooks/
│   ├── 2_blog_agent_with_updated_prompt.ipynb
│   ├── 3_blog_agent_with_research.ipynb
│   └── 4_blog_agent_with_image_gen.ipynb
│
├── static/
│   ├── CSS/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── templates/
│   └── index.html
│
├── app.py
├── backend.py
├── README.md
├── requirements.txt
├── .gitignore
└── .env
```

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI application, SSE workflow execution, downloads, and health endpoint |
| `backend.py` | LangGraph nodes, research, planning, workers, reducer, image generation, and PostgreSQL checkpointing |
| `templates/index.html` | AgentWriter AI web interface |
| `static/CSS/style.css` | Red-neon application styling |
| `static/js/app.js` | Frontend workflow streaming and interactions |
| `notebooks/` | Development and learning iterations of the blog-agent workflow |
| `assets/screenshots/` | README application and output screenshots |
| `assets/examples/example.md` | Example article generated by AgentWriter AI |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Local and generated files excluded from Git |
| `.env` | Local API keys and database configuration |

The `.env`, virtual environment, runtime `outputs/`, runtime `images/`, and unwanted generated root Markdown files should remain excluded from Git.

---

<a id="installation"></a>

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sudeeraattanayake/AgentWriter-AI.git
cd AgentWriter-AI
```

> If your GitHub repository slug is different, replace `AgentWriter-AI` in the clone URL and `cd` command with the actual repository name.

### 2️⃣ Create a Python 3.11 Virtual Environment

#### Windows PowerShell

```powershell
py -3.11 -m venv agentwriter
.\agentwriter\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3.11 -m venv agentwriter
source agentwriter/bin/activate
```

### 3️⃣ Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Configure Environment Variables

Create a `.env` file beside `app.py`:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
DATABASE_URL=your_postgresql_connection_url_here
```

Do not commit real API keys or database credentials.

### 6️⃣ Run AgentWriter AI

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

For development with automatic reload:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

---

## 🔐 Environment Configuration

Environment variables are loaded using `python-dotenv`:

```python
from dotenv import load_dotenv

load_dotenv()
```

### Required Configuration

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Authenticate OpenAI language and image requests |
| `TAVILY_API_KEY` | Authenticate Tavily research requests |
| `DATABASE_URL` | Connect the LangGraph PostgreSQL checkpointer |

Example:

```dotenv
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
DATABASE_URL=postgresql://user:password@host:5432/database
```

**Never commit real credentials to GitHub.**

An optional `.env.example` can contain placeholders only.

---

## 🌐 FastAPI Endpoints

### Home

```http
GET /
```

Serves the AgentWriter AI interface through Jinja2.

### Run AgentWriter Workflow

```http
POST /api/run
```

Starts an AgentWriter workflow run and streams execution events to the browser.

### Health Check

```http
GET /api/health
```

Used by the frontend to display application/server availability.

### Download Generated Article

```http
GET /api/runs/{run_id}/download
```

Returns the generated Markdown article associated with a completed run when the output exists.

---

## 🎨 Frontend and Animations

AgentWriter AI uses a custom dark interface with red-neon styling.

### Red-Neon Workspace

The interface includes:

- Deep black/red application background.
- Neon-red glow effects.
- 3D-style cards and interface depth.
- Animated AgentWriter branding.
- Workflow-stage navigation.
- Server-online indicator.
- Animated progress visualization.
- Execution timeline.
- Strategy/article-plan panel.
- Generated-article workspace.
- Responsive layout.

### Execution Experience

While the graph runs, the frontend reflects stages such as:

```text
Understanding request...
Researching sources...
Creating article plan...
Writing sections...
Generating visuals...
Finalizing article...
```

These interface updates are driven by workflow events emitted from the FastAPI application.

### Markdown Rendering

The frontend uses **Marked** to transform generated Markdown into rendered article HTML.

### Safe HTML Rendering

**DOMPurify** sanitizes rendered Markdown HTML before it is inserted into the article preview.

### Syntax Highlighting

**Highlight.js** detects generated code blocks and applies language-aware syntax highlighting inside the article preview.

### Responsive Design

The interface adapts to smaller desktop, tablet, and mobile layouts.

---

## 📥 Generated Outputs

### Markdown Article

The final result is generated as Markdown and can be saved/downloaded as a `.md` file.

### Runtime Images

When the image-planning stage requests visuals, generated images are written to the runtime image directory and referenced from the Markdown article.

### Repository Example

A curated generated article is stored separately from runtime outputs:

```text
assets/examples/example.md
```

This allows visitors to inspect a real AgentWriter AI result directly on GitHub without committing every generated article.

**[📄 Read the example generated article](assets/examples/example.md)**

---

## 💬 Example Prompts

### 🦜 LangGraph

> Explain LangGraph subgraphs with a practical multi-agent example, code, architecture, debugging guidance, and production considerations.

### 🔎 Retrieval-Augmented Generation

> Design a production-ready RAG system with PostgreSQL and LangGraph, including ingestion, retrieval, evaluation, observability, security, and failure modes.

### 🤖 Agentic AI

> Write a technical guide to building a production multi-agent AI system with LangGraph, including routing, parallel workers, reducers, persistence, and human-in-the-loop patterns.

### 🧠 Transformers

> Explain self-attention in Transformer architecture from the mathematics to a minimal working implementation and common debugging mistakes.

### 🏗️ System Design

> Compare different architectures for production LLM applications and explain reliability, latency, cost, observability, and scaling trade-offs.

### 📰 Fresh AI Topic

> Create a current roundup of major multimodal LLM developments and explain what they mean for application developers.

---

## 🧩 Error Handling

AgentWriter AI includes validation and defensive handling across research, generation, persistence, and image creation.

Possible failures include:

- Missing OpenAI credentials.
- Missing Tavily credentials.
- Invalid PostgreSQL connection configuration.
- Network or provider failures.
- OpenAI model-access errors.
- Tavily response-shape differences.
- Empty research results.
- Image-generation failures.
- PostgreSQL checkpoint errors.
- Invalid generated filenames.
- Malformed or unexpected external search results.
- Null characters in generated model content.

### Tavily Response Normalization

The research helper accepts supported dictionary/list response shapes and skips malformed non-dictionary results.

### Image Failure Handling

If an image cannot be generated, AgentWriter AI can replace the placeholder with a Markdown failure block containing relevant context rather than terminating article assembly.

### PostgreSQL Null-Character Protection

Generated strings are sanitized to remove:

```text
\u0000
```

before relevant state is persisted because PostgreSQL text/JSON values cannot contain the null character.

---

## ⚠️ Current Limitations

- Generated technical content can contain inaccuracies and should be reviewed before production use.
- Research quality depends on the search results returned by Tavily.
- Citation quality depends on the evidence supplied to worker agents.
- The research stage does not guarantee exhaustive coverage of every source.
- Image generation depends on OpenAI model access and API availability.
- AI-generated diagrams can contain imperfect labels or technical details.
- PostgreSQL availability is required when running with the configured persistent checkpointer.
- The current interface does not provide user authentication.
- Generated runtime articles and images are local application outputs unless separately deployed or stored.
- The application does not currently provide collaborative editing.
- Article generation can take time because planning, research, multiple worker calls, reduction, and image generation may all execute during one run.
- Public deployment would require additional authentication, authorization, rate limiting, secrets management, monitoring, and production concurrency controls.

---

## 🎯 Project Objectives

- Build an end-to-end agentic technical-writing application.
- Design conditional research routing using LangGraph.
- Separate routing, research, planning, writing, reduction, and image generation.
- Use structured outputs for reliable agent-to-agent data transfer.
- Implement parallel section generation with `Send`.
- Combine parallel worker results with a reducer.
- Integrate Tavily for external research.
- Support evidence-aware technical content.
- Generate useful technical visuals through OpenAI.
- Persist LangGraph checkpoints with PostgreSQL.
- Stream workflow progress to a browser in real time.
- Build a FastAPI application around the agent workflow.
- Connect the backend to a custom HTML, CSS, and JavaScript frontend.
- Render Markdown and highlighted code.
- Produce downloadable `.md` articles.
- Demonstrate practical multi-agent AI engineering through a portfolio project.

---

## 📚 Learning Areas

### 🐍 Python

- Typed application state.
- Pydantic schemas.
- File handling.
- Environment-variable configuration.
- PostgreSQL connections.
- Exception handling.
- Recursive data sanitization.
- Modular application development.

### 🦜 LangChain and LangGraph

- `StateGraph`.
- Conditional routing.
- `Send` fan-out.
- Reducers with `Annotated`.
- Structured outputs.
- Subgraphs.
- Shared graph state.
- PostgreSQL checkpointing.
- Multi-stage agent workflows.

### 🤖 Large Language Models

- System prompting.
- Structured routing.
- Evidence synthesis.
- Article planning.
- Section generation.
- Grounded writing.
- Citation-aware prompting.
- Image planning.
- Multi-stage generation.

### 🔎 Web Research

- Tavily integration.
- Search-query generation.
- Search-result normalization.
- Evidence extraction.
- URL deduplication.
- Fresh vs. evergreen content routing.

### 🐘 Persistence

- PostgreSQL connectivity.
- SSL database configuration.
- `PostgresSaver`.
- LangGraph checkpoint setup.
- Persistent thread state.
- Database-safe content sanitization.

### 🌐 FastAPI

- HTTP routes.
- Jinja2 templates.
- Static-file serving.
- SSE streaming.
- Health endpoints.
- File downloads.
- Error propagation.

### 🎨 Frontend Development

- HTML interface development.
- Red-neon CSS design.
- Responsive layouts.
- JavaScript Fetch API.
- SSE parsing.
- DOM updates.
- Markdown rendering.
- HTML sanitization.
- Syntax highlighting.
- Real-time workflow visualization.

---

## 🚀 Future Improvements

- [ ] Add user authentication.
- [ ] Add saved article history.
- [ ] Add article editing before download.
- [ ] Add configurable audience and tone controls.
- [ ] Add selectable article length.
- [ ] Add source-selection controls.
- [ ] Add stronger citation verification.
- [ ] Add source-quality scoring.
- [ ] Add article revision agents.
- [ ] Add critic/reviewer feedback loops.
- [ ] Add human-in-the-loop article approval.
- [ ] Add guardrails for generated technical claims.
- [ ] Add configurable image-generation controls.
- [ ] Add persistent storage for generated articles and images.
- [ ] Add automated unit and integration tests.
- [ ] Add LangSmith tracing and evaluation.
- [ ] Add token, latency, and API-cost monitoring.
- [ ] Add Docker containerization.
- [ ] Add CI/CD.
- [ ] Add production deployment documentation.
- [ ] Add production monitoring and alerting.

These are proposed extensions and are not claims about the current implementation.

---

## 🔒 Configuration Hygiene

Keep credentials, virtual environments, generated runtime files, caches, and local configuration out of version control:

```gitignore
# Environment
.env

# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
agentwriter/
venv/
.venv/

# Generated runtime content
outputs/
images/

# Generated notebook images
notebooks/images/

# Generated notebook articles
notebooks/*.md

# IDE
.vscode/

# OS
.DS_Store
Thumbs.db
```

Keep the curated example article under:

```text
assets/examples/example.md
```

so it can remain visible in the repository while normal runtime output stays ignored.

Never commit:

```text
OPENAI_API_KEY
TAVILY_API_KEY
DATABASE_URL
```

If a credential has already been committed, adding it to `.gitignore` does not remove it from Git history. Revoke or rotate the exposed credential.

---

## ⭐ Project Highlights

### 🤖 Multi-Agent Technical Writing

AgentWriter AI separates technical content generation into specialized routing, research, planning, writing, reduction, and visual-generation stages.

### 🧭 Adaptive Research Routing

The Router determines whether an article can be generated from evergreen knowledge or should use external web evidence.

### 🔎 Research-Grounded Generation

Tavily evidence can be incorporated into article planning and citation-aware worker prompts.

### ⚡ Parallel Writer Architecture

LangGraph `Send` distributes planned sections to worker executions, while the reducer restores deterministic article order.

### 🧠 Structured Article Strategy

Pydantic schemas provide explicit goals, bullets, word targets, tags, research flags, citation requirements, and code requirements.

### 🔀 Reducer Subgraph

Article merging and visual generation are encapsulated in a dedicated LangGraph subgraph.

### 🖼️ AI-Generated Technical Visuals

The image-planning stage selectively requests architecture diagrams, flows, or educational illustrations when they improve understanding.

### 🐘 Persistent Agent Workflow

PostgreSQL `PostgresSaver` provides checkpoint persistence for the compiled LangGraph workflow.

### 📡 Real-Time Execution Visibility

SSE lets the frontend show agent activity and workflow progress while generation is still running.

### 🌐 Full-Stack AI Application

FastAPI connects the LangGraph backend to a custom HTML, CSS, and JavaScript interface.

### 🎨 Red-Neon Interface

The frontend combines dark styling, neon-red effects, 3D depth, workflow visualization, article planning, and rendered technical output.

### 📄 Real Generated Blog Example

The repository includes a complete generated Markdown article so visitors can inspect the actual output rather than only application screenshots.

### 🧩 Modular Architecture

Research, orchestration, workers, reducer logic, image generation, persistence, API routes, frontend markup, styling, and JavaScript behavior remain separated by responsibility.

---

## 📌 Repository

**AgentWriter AI — Real-Time Multi-Agent Technical Content Generation Platform**

> Add your final GitHub repository URL here if the repository slug differs from https://github.com/sudeeraattanayake/AgentWriter-AI.

---

## 👨‍💻 Author

**Sudeera Attanayake**

Generative AI | LLM Applications | AI Agents | Python

**GitHub:** https://github.com/sudeeraattanayake

---

## 🤝 Support

If you find this project useful:

- ⭐ Star the repository.
- 🐛 Report issues.
- 💡 Suggest improvements.
- 📚 Explore the implementation.
- 📝 Read the generated article example.

---

<p align="center">
  <strong>
    Built with 🐍 Python + 🦜 LangGraph + 🤖 OpenAI + 🔎 Tavily + 🐘 PostgreSQL + 🌐 FastAPI + 📡 SSE
  </strong>
</p>

<p align="center">
  <strong>✦ AgentWriter AI — Research. Plan. Write. Visualize.</strong>
</p>
