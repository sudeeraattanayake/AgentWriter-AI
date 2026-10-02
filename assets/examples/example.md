Blog kind: tutorial
Topic: Explain LangGraph subgraphs with a practical multi-agent example

# LangGraph Subgraphs: A Practical Multi-Agent Example

## What is a LangGraph subgraph — concepts and when to use one

Goal: Define what a LangGraph subgraph is and give concrete decision criteria for when to extract functionality into one.

- Definition: a subgraph is a self-contained LangGraph that groups multiple nodes and edges into a single reusable unit. It exposes a clear input/output interface, can hold internal (scoped) state, and encapsulates control flow and transformations. Unlike a single node/action, a subgraph packages multiple steps, branching, and state transitions so you can treat a complex workflow as one composable block for reuse and abstraction.

- Concrete use cases: isolate agent decision logic (handoffs, prompts), encapsulate and reuse complex workflows (data prep → model → postprocess), wrap parallel or branching logic (map/reduce, fan-out/fan-in), and create sandboxed components for unit testing or safe experimentation.

- Trade-offs: + Improved reuse and clearer parent graphs. − Compile/serialize overhead for subgraphs, extra runtime indirection, and potentially harder-to-debug nested failures. Also be mindful of state scoping: local state is safer but adds synchronization if parents need visibility.

- Extraction checklist: single responsibility, repeated usage across graphs, independent/encapsulated state, or non-trivial control flow (forks/joins).

- Testing & CI: compile and test subgraphs independently as unit targets; smaller integration surface reduces CI flakiness and speeds up end-to-end pipeline tests.

## Anatomy of a LangGraph subgraph (compiled subgraph vs node-action)

Goal: Break down the parts of a subgraph and explain the two common invocation styles: compiled subgraph and subgraph-as-node-action.

- Internal pieces (what a subgraph contains): an entry node, one or more exit nodes, explicitly typed inputs/outputs, a state schema that distinguishes scoped vs global values, and optional checkpointer interactions for resuming or persisting progress. See the LangGraph reference and docs for the canonical input/output and state concepts ([LangGraph Reference](https://reference.langchain.com/python/langgraph), [Use Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)).

- Compiled subgraph vs subgraph-as-node-action (lifecycle & serialization): a compiled subgraph is a first-class Graph object you build, validate, serialize, and store independently; it has its own lifecycle, metadata, and can be versioned ([compiled example](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-compiledgraph.ipynb)). A subgraph-as-node-action is embedded as a node implementation invoked at runtime and usually serializes only the node config; it’s lighter-weight but less discoverable ([node-action example](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-nodeaction.ipynb)).

![Diagram comparing Compiled Subgraph vs Subgraph-as-Node-Action](images/compiled_vs_nodeaction.png)

*Side-by-side technical comparison: compiled subgraph (first-class Graph, versioned, serializable) vs subgraph-as-node-action (embedded node, light-weight runtime invocation).*

- Interface contracts: parents pass inputs by stable keys and schemas and expect outputs at agreed exit keys; keeping input/output keys and types stable is critical to avoid runtime mismatches (see docs and examples) ([Use Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)).

- Execution primitives & control flow: jumps, interrupts, and fork/merge behave differently when enclosed—interrupts typically bubble to the subgraph boundary unless handled inside; forks may require explicit coordination or external orchestration due to limited parallel node support ([subgraph interruption demo](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption), discussion on branching workarounds ([discussion #371](https://github.com/langgraph4j/langgraph4j/discussions/371))).

- Concrete resources to inspect implementations: compiled vs node-action notebooks, interruption reference, multi-agent samples and tutorials ([compiled notebook](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-compiledgraph.ipynb), [node-action notebook](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-nodeaction.ipynb), [subgraph interruption](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption), [aws multi-agent sample](https://github.com/aws-samples/langgraph-multi-agent)).

## Designing the multi-agent example: agents, responsibilities, and subgraph boundaries

Goal: Define a concrete multi-agent scenario and show how each agent becomes a LangGraph subgraph, with explicit responsibilities, data passed, and communication patterns.  
*(This section uses fresh repo/docs/blog info — requires_research, requires_citations.)*

- Outline the scenario: Planner, Data Agent, Executor — Planner breaks a user goal into ordered steps and metadata (step id, prompt template, expected inputs); Data Agent fetches/preprocesses documents or API results and returns normalized payloads; Executor runs each step (LLM call, tool invocation), captures outputs and status, and aggregates results into a final report. This split mirrors common multi-agent patterns in LangGraph examples and tutorials ([Use Subgraphs — LangChain Docs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs), [langgraph4j how-tos](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-compiledgraph.ipynb)). 

- Map agents to subgraphs: Planner subgraph contains planning logic, task decomposition, and validation; Data Agent subgraph encapsulates connectors, caching, and normalization; Executor subgraph holds execution logic, retries, and aggregation. Passed values: task list, task metadata, normalized data pointers, execution results. Keep state scoped (current task, temp cache) inside subgraphs; expose only explicit outputs (task list, result summary) to a global orchestrator subgraph ([Use Subgraphs — LangChain Docs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)).

![Multi-agent architecture: Planner, Data Agent, Executor subgraphs and handoffs](images/multi_agent_subgraphs.png)

*Architecture diagram showing Planner, Data Agent, and Executor as separate subgraphs with synchronous and asynchronous handoff patterns and data passed between them.*

- Sketch interaction patterns: synchronous call/return for planner→executor when immediate steps are small; async invocation (fire-and-forget or job queue) for long-running Data Agent fetches; handoff patterns where Planner delegates a step to Data Agent or Executor and awaits a callback/interruptible subgraph completion (see Agents Handoff and subgraph interruption examples) ([Agents Handoff](https://github.com/langgraph4j/langgraph4j/discussions/217), [subgraph-interruption](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption)).

- Call out reusable components: implement shared utility subgraphs for retry/backoff, rate-limited connectors, data normalization, and schema validators so multiple agents invoke the same tested logic (patterns discussed in community and docs) ([Poor support for parallel/fork nodes](https://github.com/langgraph4j/langgraph4j/discussions/371), [Use Subgraphs — LangChain Docs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)).

- Reference example repos and community patterns to inspect concrete graphs and notebooks: langgraph4j how-tos and node-action examples, langgraph4j-examples/subgraph-interruption, aws-samples/langgraph-multi-agent, LangGraph-learn, and broader tutorials — review these for notebooks, concrete graph definitions, and discussion threads ([langgraph4j how-tos](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-nodeaction.ipynb), [subgraph-as-compiledgraph](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-compiledgraph.ipynb), [langgraph4j-examples subgraph-interruption](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption), [aws-samples/langgraph-multi-agent](https://github.com/aws-samples/langgraph-multi-agent), [LangGraph-learn](https://github.com/LangGraph-GUI/LangGraph-learn)).

## Minimal working example: implement a compiled subgraph and call it from a parent graph (Python)

Goal: Provide a compact, runnable Python sketch that compiles a subgraph, registers it in a parent LangGraph, runs the workflow, and verifies outputs.

- Environment and setup
  - Install the Python packages and clone example repos used for reference:
    - pip install langchain langgraph openai pytest  # (adjust package names if your environment differs) ([LangGraph reference](https://reference.langchain.com/python/langgraph))  
    - Optional example repos to inspect for patterns and tests:
      - git clone https://github.com/langgraph4j/langgraph4j.git ([subgraph examples](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-compiledgraph.ipynb))  
      - git clone https://github.com/langgraph4j/langgraph4j-examples.git ([subgraph interruption example](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption))  
      - git clone https://github.com/aws-samples/langgraph-multi-agent.git (multi-agent patterns)  
    - Runtime config: set your LLM provider API key as an env var, e.g. export OPENAI_API_KEY="sk-..." (or other provider key expected by your LangGraph runtime). See the LangChain docs for provider setup ([Use Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs), [LangGraph reference](https://reference.langchain.com/python/langgraph)).
  - Note: the example repos linked above show canonical patterns for compiled subgraphs and node-as-action registration; consult those for exact API versions ([compiled subgraph how‑to](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-compiledgraph.ipynb)).

- Create the subgraph
  - Define a tiny subgraph with 2 nodes: (1) accept an input string, (2) transform it (uppercase + append suffix), then return the transformed value.
  - Compile that subgraph to a reusable artifact (a "compiled subgraph") you can inject into parent graphs.
  - Minimal runnable sketch (adapt names to your installed langgraph API; this is a compact, real Python pattern you can adapt):
```python
# file: subgraph_mwe.py
from langchain.langgraph import Graph, Node  # adapt import path to your langgraph package

# Node that accepts payload and returns transformed string
def transform_fn(inputs: dict):
    s = inputs["text"]
    return {"transformed": s.upper() + " -- processed by subgraph"}

# Build and compile subgraph
sub = Graph(name="simple_subgraph")
sub.add_node(Node("transform", func=transform_fn, inputs=["text"], outputs=["transformed"]))
compiled_sub = sub.compile()  # produces a compiled subgraph artifact
```

- Register and invoke
  - Add the compiled subgraph into a parent graph as a node/node-action, wire inputs/outputs, then run with a sample payload.
  - Example parent graph and invocation:
```python
# file: parent_run.py
from langchain.langgraph import Graph, Node
from subgraph_mwe import compiled_sub

parent = Graph(name="parent_graph")
# register compiled subgraph as a node/action named "sub_call"
parent.add_node(compiled_sub.as_node("sub_call", inputs=["text"], outputs=["transformed"]))
# a passthrough node to collect/substitute or further process the subgraph output
parent.add_node(Node("collector", func=lambda i: {"result": i["transformed"]}, inputs=["transformed"], outputs=["result"]))
parent.connect("sub_call", "collector", ["transformed"])
res = parent.run({"text": "hello world"})
print(res)  # expect {'result': 'HELLO WORLD -- processed by subgraph'}
```

![Flow of parent graph calling compiled subgraph with internal transform node](images/parent_subgraph_flow.png)

*Flowchart for the minimal working example: Parent Graph invokes compiled subgraph (transform node) and a collector node aggregates the transformed output.*

  - See patterns for subgraph-as-nodeaction in the langgraph4j how‑to ([subgraph-as-nodeaction](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-nodeaction.ipynb)).

- Verification
  - Run the parent graph and assert expected outputs; add quick unit tests to validate state passed to/from the subgraph:
```python
# file: test_mwe.py
from parent_run import parent  # or import a run() helper that returns the result

def test_subgraph_roundtrip():
    out = parent.run({"text": "test"})
    assert out["result"] == "TEST -- processed by subgraph"
```
  - You can also assert intermediate state (if the runtime exposes node-level state or logs) to ensure inputs were forwarded correctly and outputs were returned by the compiled subgraph (see interruption and debugging examples in the langgraph4j examples repo ([subgraph interruption](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption))).

- Tips to adapt (multi‑agent / production)
  - Swap simple function nodes with agent calls (LLM agents or tool-enabled agents) to turn the subgraph into an agent microservice; the parent graph merely triggers the compiled subgraph node/action. See multi-agent examples and handoff discussions ([aws-samples/langgraph-multi-agent](https://github.com/aws-samples/langgraph-multi-agent), [Agents Handoff discussion](https://github.com/langgraph4j/langgraph4j/discussions/217)).
  - Add retries/timeouts on the parent → subgraph edge or inside the subgraph node to handle transient LLM errors (patterns shown in examples and forum discussions).
  - For full API details and advanced features (dynamic subgraphs, parallel/fork patterns), consult the LangGraph docs and community how‑tos linked above ([LangChain docs Use Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs), reference page ([LangGraph reference](https://reference.langchain.com/python/langgraph))).

## State, handoff patterns, and edge cases to watch

Goal: Show how to pass execution state between parent and subgraphs, implement agent handoffs safely, and enumerate common failure modes.  
(requires_research: true, requires_citations: true)

- State patterns: scoped values vs. global context
  - Use Scoped Values for short-lived, typed inputs/outputs that live only for a single call/agent (safer, avoids namespace pollution). For cross-call or audit needs, serialize a typed global context object and pass its reference into the subgraph. ([Source](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-nodeaction.ipynb), [Source](https://docs.langchain.com/oss/python/langgraph/use-subgraphs))

- How to pass typed inputs/outputs and when to checkpoint externally
  - Define explicit input/output schemas on the subgraph interface so runtime validates types at entry/exit. Checkpoint externally whenever you perform non-idempotent actions (writes, API calls) or when workflows span retries or human intervenions. ([Source](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-compiledgraph.ipynb), [Source](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption))

- Agent handoff patterns
  - Pass execution context as a small immutable bundle (Scoped Values + minimal global ref). Use an explicit "handoff" node that packages context, invokes the subgraph, and exposes an "exit" node so the subgraph returns control deterministically. See community handoff patterns for practical examples. ([Source](https://github.com/langgraph4j/langgraph4j/discussions/217), [Source](https://github.com/aws-samples/langgraph-multi-agent))

- Edge cases and failure modes
  - Watch for state name collisions, partial updates that leave inconsistent state, schema drift between parent/subgraph, and non-idempotent side effects that cannot be rolled back. These show up in interruption examples and branching discussions. ([Source](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption), [Source](https://github.com/langgraph4j/langgraph4j/discussions/371))

- Recovery strategies and debugging tips
  - Validate inputs at subgraph entry, snapshot state before risky ops, and use retries or compensating actions on failure. Add lightweight invariant/assert nodes at boundaries, log state diffs on entry/exit, and write small unit tests for compiled subgraphs to exercise schemas. ([Source](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption), [Source](https://docs.langchain.com/oss/python/langgraph/use-subgraphs), [Source](https://github.com/LangGraph-GUI/LangGraph-learn))

## Debugging, observability, and testing strategies for subgraphs

Goal: Practical, repeatable steps to observe subgraph execution, reproduce failures, and test subgraphs both in isolation and inside the parent graph.

- Logging points — where to instrument
  - Log entry/exit of every subgraph invocation and record caller/subgraph IDs to correlate traces.
  - Add node-level logs for start/finish, decision branches, and error stacks; capture structured inputs/outputs (JSON) so traces are queryable.
  - Prefer structured logging with keys like run_id, parent_id, node_id, and step to enable cross-run queries. (See LangChain docs/use-subgraphs and langgraph-101 for patterns) ([Source](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)) ([Source](https://github.com/langchain-ai/langgraph-101))

- Checkpointers and execution traces
  - Persist compiled graph artifacts and their metadata; inspect these artifacts to verify node ordering and I/O contracts.
  - Replay past executions from stored traces or compiled artifacts to reproduce bugs deterministically. (See subgraph-as-compiledgraph how-to) ([Source](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-compiledgraph.ipynb))

- Unit and integration testing
  - Run compiled subgraphs locally with mocked LLM responses (inject deterministic outputs), assert expected state transitions, and fail fast on schema changes.
  - Use snapshot tests for serialized graph outputs and intermediate state to detect regressions. (See LangGraph learn examples and node-action how-to) ([Source](https://github.com/LangGraph-GUI/LangGraph-learn)) ([Source](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-nodeaction.ipynb))

- Reproducing interruptions
  - Simulate node failures (raise errors, kill process, or return timeouts) and replay runs to verify parent/subgraph recovery and checkpoint resumption.
  - Use the subgraph-interruption examples/notebooks to follow concrete interruption scenarios and recovery tests. ([Source](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption))

- Observability checklist and quick queries
  - Capture: latency per node/subgraph, token usage, state sizes (payload bytes), error rates, and run ancestry (parent↔subgraph).
  - Correlate traces using run_id/parent_id; query for high-latency runs, excessive token consumption, or repeated retries.
  - Quick queries: "SELECT runs WHERE latency>1500ms ORDER BY token_usage DESC" and "find runs with >N restarts for same parent_id" — adapt to your observability backend. (See LangChain reference and discussions on branching/workflows) ([Source](https://reference.langchain.com/python/langgraph)) ([Source](https://github.com/langgraph4j/langgraph4j/discussions/371))

## Performance, scaling, and cost trade-offs for subgraph-based multi-agent systems

Goal: Explain how subgraphs affect performance and cost, and give concrete strategies to scale multi-agent LangGraph workflows efficiently. (requires_research, requires_citations)

- Parallelization and forks — trade-offs when wrapping parallel branches into subgraphs:
  - Wrapping each branch in its own subgraph simplifies logic and failure isolation, but increases scheduler overhead and can limit true concurrency if the runtime has weak fork support; LangGraph runtimes and community discussions note limitations around fork/parallel-node handling you should design around ([Source](https://github.com/langgraph4j/langgraph4j/discussions/371)).  
  - Use subgraphs when isolation (retries, timeouts) matters; keep very short-lived parallel tasks inline to avoid pay/per-call overhead and orchestration latency (see subgraph interruption patterns for examples) ([Source](https://github.com/langgraph4j/langgraph4j-examples/blob/main/reference/subgraph-interruption)).

- Measure and limit costs:
  - Instrument each subgraph for LLM calls and token usage (requests, prompt + response tokens) so you can roll up cost per-subgraph and per-workflow; tie these metrics to run IDs to attribute costs. Docs and how-tos show patterns for treating subgraphs as measurable units ([Source](https://github.com/langgraph4j/langgraph4j/blob/main/how-tos/subgraph-as-compiledgraph.ipynb)).  
  - Batch similar LLM calls (map inputs → single batched prompt) where semantic correctness allows, cache deterministic responses, and push cheap preprocessing/local heuristics outside the LLM to avoid needless invocations ([Source](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)).

- Scaling strategies:
  - Make embarrassingly parallel tasks stateless subgraphs so horizontal scaling is straightforward; pack heavy I/O (DB/file access) into dedicated subgraphs to avoid blocking LLM-heavy paths ([Source](https://github.com/aws-samples/langgraph-multi-agent)).  
  - Prefer asynchronous execution and non-blocking runtimes where supported; isolate long-running or blocking steps into separate workers to keep throughput high ([Source](https://www.langchain.com/blog/langgraph-multi-agent-workflows)).

- Benchmarking tips:
  - Capture latency (p50/p95/p99), tokens per run, LLM calls, throughput (runs/sec), and memory/CPU during runs. Track cost per run (tokens × pricing) alongside these metrics ([Source](https://aipractitioner.substack.com/p/scaling-langgraph-agents-parallelization)).  
  - Experiment designs: (1) fix input size, vary concurrency to find saturation; (2) fix concurrency, vary input size to probe nonlinear token growth; (3) compare single big subgraph vs broken-up subgraphs to measure orchestration overhead. Interpret tipping points where latency or cost grows superlinearly as signs to change granularity.

- Architectural trade-offs:
  - Shared state reduces serialization/checkpoint overhead but increases coupling and memory footprint; isolated subgraphs simplify recovery and scaling but cost more in orchestration and storage for checkpoints ([Source](https://github.com/langchain-ai/langgraph-101)).  
  - Prefer simple parent-level coordination (few subgraphs) when logic is lightweight or tightly coupled; split into many tiny subgraphs when you need independent retries, different scaling profiles, or strong isolation of secrets/IO.