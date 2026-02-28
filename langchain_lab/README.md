# LangChain & LangGraph: Complete Guide for PE-OrgAIR

> **Course:** SpringBigData — Graduate Data Engineering  
> **Platform:** PE-OrgAIR (Private Equity Organizational AI Readiness)  
> **Purpose:** LLM-powered intelligence layer replacing keyword-based analysis

---

## Table of Contents

1. [What is LangChain?](#1-what-is-langchain)
2. [What is LangGraph?](#2-what-is-langgraph)
3. [LangChain vs LangGraph — When to Use Which](#3-langchain-vs-langgraph--when-to-use-which)
4. [Core LangChain Concepts](#4-core-langchain-concepts)
5. [Core LangGraph Concepts](#5-core-langgraph-concepts)
6. [PE-OrgAIR Integration Examples](#6-pe-orgair-integration-examples)
7. [Architecture Patterns](#7-architecture-patterns)
8. [Quick Reference](#8-quick-reference)

---

## 1. What is LangChain?

**LangChain** is a framework for building applications powered by Large Language Models (LLMs). It provides a standard interface for chaining together components like prompts, models, parsers, and tools into **pipelines** called **chains**.

### Core Idea: Chain = Prompt → LLM → Output Parser

```
Input → PromptTemplate → LLM → OutputParser → Structured Result
```

### Key Features

| Feature | Description |
|---|---|
| **Chains** | Sequential pipelines: A → B → C |
| **Prompt Templates** | Reusable, parameterized prompts |
| **Output Parsers** | Convert LLM text into structured Python objects |
| **Memory** | Persist context across conversation turns |
| **Tools** | Connect LLMs to external APIs, databases, search |
| **Retrievers** | Fetch relevant documents for RAG (Retrieval-Augmented Generation) |

### Installation

```bash
pip install langchain langchain-openai langchain-anthropic
```

### Minimal Example

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Define the prompt
prompt = ChatPromptTemplate.from_template(
    "Analyze the AI readiness of {company_name} based on: {filing_text}"
)

# 2. Define the model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Define the output parser
parser = StrOutputParser()

# 4. Build the chain using the | (pipe) operator
chain = prompt | llm | parser

# 5. Invoke the chain
result = chain.invoke({
    "company_name": "TechCorp",
    "filing_text": "We invested $50M in AI infrastructure..."
})
print(result)
```

---

## 2. What is LangGraph?

**LangGraph** is an extension of LangChain for building **stateful, multi-actor applications** using a **graph-based execution model**. Instead of linear chains, LangGraph lets you define workflows as **nodes** and **edges** in a directed graph — with support for **loops**, **conditional branching**, and **human-in-the-loop** checkpoints.

### Core Idea: Graph = Nodes (functions) + Edges (transitions) + State (shared memory)

```
         ┌──────────────────────────────────────────┐
         │              StateGraph                  │
         │                                          │
  START ──► fetch_filings ──► analyze_dimensions    │
         │        │                  │              │
         │        │          ┌───────┴──────┐       │
         │        │          ▼              ▼       │
         │        │    score_high?    needs_review? │
         │        │          │              │       │
         │        │          ▼              ▼       │
         │        │      generate_report  human_check│
         │        │          │              │       │
         │        └──────────┴──────────────┘       │
         │                   │                      │
         │                  END                     │
         └──────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---|---|
| **StateGraph** | Graph where each node reads/writes shared state |
| **Nodes** | Python functions or LLM calls |
| **Edges** | Fixed transitions between nodes |
| **Conditional Edges** | Branch based on state values |
| **Cycles / Loops** | Retry logic, iterative refinement |
| **Checkpointing** | Pause, resume, inspect state at any point |
| **Human-in-the-Loop** | Interrupt graph for human approval (HITL) |
| **Streaming** | Stream tokens/events from intermediate nodes |

### Installation

```bash
pip install langgraph
```

### Minimal Example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. Define shared state
class AIReadinessState(TypedDict):
    company_name: str
    filing_text: str
    dimensions: dict
    final_score: float
    needs_review: bool

# 2. Define node functions
def analyze_filing(state: AIReadinessState) -> AIReadinessState:
    # Call LLM to extract AI signals
    state["dimensions"] = {"data_infrastructure": 0.8, "talent": 0.6}
    return state

def calculate_score(state: AIReadinessState) -> AIReadinessState:
    scores = list(state["dimensions"].values())
    state["final_score"] = sum(scores) / len(scores)
    state["needs_review"] = state["final_score"] < 0.5
    return state

def generate_report(state: AIReadinessState) -> AIReadinessState:
    print(f"Report for {state['company_name']}: Score = {state['final_score']:.2f}")
    return state

# 3. Define conditional routing
def route_after_scoring(state: AIReadinessState) -> str:
    return "human_review" if state["needs_review"] else "generate_report"

# 4. Build the graph
graph = StateGraph(AIReadinessState)
graph.add_node("analyze_filing", analyze_filing)
graph.add_node("calculate_score", calculate_score)
graph.add_node("generate_report", generate_report)

graph.set_entry_point("analyze_filing")
graph.add_edge("analyze_filing", "calculate_score")
graph.add_conditional_edges("calculate_score", route_after_scoring, {
    "human_review": END,           # Pause for HITL
    "generate_report": "generate_report"
})
graph.add_edge("generate_report", END)

# 5. Compile and run
app = graph.compile()
result = app.invoke({
    "company_name": "TechCorp",
    "filing_text": "...",
    "dimensions": {},
    "final_score": 0.0,
    "needs_review": False
})
```

---

## 3. LangChain vs LangGraph — When to Use Which

| Criteria | LangChain (Chains) | LangGraph (Graphs) |
|---|---|---|
| **Execution Flow** | Linear: A → B → C | Graph: can loop, branch, cycle |
| **State Management** | No built-in shared state | Typed shared state object |
| **Complexity** | Simple to medium pipelines | Complex multi-step agents |
| **Branching** | Limited (RunnableBranch) | First-class conditional edges |
| **Loops / Retries** | Not native | Native cycles supported |
| **Human-in-the-Loop** | Manual implementation | Built-in interrupt/checkpoint |
| **Streaming** | Token-level streaming | Node-level + token streaming |
| **Best For** | Single-pass analysis, Q&A, RAG | Multi-agent, iterative workflows |

### Decision Rule for PE-OrgAIR

```
Need to analyze ONE filing → LangChain Chain
Need to orchestrate MULTIPLE agents with routing → LangGraph
Need Human-in-the-Loop approval → LangGraph (HITL threshold)
Need retry/reflection loops → LangGraph
```

---

## 4. Core LangChain Concepts

### 4.1 Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

# System + Human message template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert PE analyst. Score companies on AI readiness."),
    ("human", "Company: {company}\nFiling excerpt: {text}\nProvide scores for all 7 dimensions.")
])
```

### 4.2 Structured Output Parsing

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class AIReadinessScores(BaseModel):
    data_infrastructure: float = Field(ge=0, le=100, description="Score 0-100")
    technology_stack: float = Field(ge=0, le=100)
    talent_capability: float = Field(ge=0, le=100)
    leadership_vision: float = Field(ge=0, le=100)
    ai_governance: float = Field(ge=0, le=100)
    use_case_portfolio: float = Field(ge=0, le=100)
    culture_change: float = Field(ge=0, le=100)
    rationale: str

# Structured output — LLM returns a typed Python object
llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(AIReadinessScores)
```

### 4.3 LCEL — LangChain Expression Language

The `|` pipe operator composes components:

```python
# Simple chain
chain = prompt | llm | output_parser

# With structured output
analysis_chain = prompt | structured_llm

# Parallel execution (RunnableParallel)
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel({
    "scores": scoring_chain,
    "summary": summary_chain,
    "risk_flags": risk_chain
})
```

### 4.4 Tools & Tool Calling

```python
from langchain_core.tools import tool

@tool
def fetch_sec_filing(ticker: str, form_type: str = "10-K") -> str:
    """Fetch SEC EDGAR filing for a company ticker."""
    # Call SEC EDGAR API
    return filing_text

@tool
def get_company_metadata(ticker: str) -> dict:
    """Get company sector, revenue, employee count from database."""
    return {"sector": "Technology", "revenue": 1_000_000_000}

# Bind tools to LLM
llm_with_tools = llm.bind_tools([fetch_sec_filing, get_company_metadata])
```

### 4.5 RAG Pattern (Retrieval-Augmented Generation)

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

# Build vector store from chunked SEC filings
vectorstore = InMemoryVectorStore.from_texts(
    texts=chunked_filings,
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 5. Core LangGraph Concepts

### 5.1 State — The Shared Memory

```python
from typing import TypedDict, Annotated, List
from operator import add  # reducer for list fields

class OrgAIRState(TypedDict):
    # Input
    ticker: str
    filing_text: str
    
    # Intermediate results
    extracted_signals: dict
    dimension_scores: Annotated[List[dict], add]  # append-only list
    
    # Output
    final_score: float
    investment_recommendation: str
    
    # Control flow
    retry_count: int
    needs_human_review: bool
    error_message: str
```

**Reducers** control how state fields are updated:
- Default: last-write wins
- `Annotated[List, add]`: append to existing list
- Custom: write your own merge function

### 5.2 Nodes — The Workers

```python
def extract_ai_signals(state: OrgAIRState) -> dict:
    """Extract AI signals from filing text using LLM."""
    response = extraction_chain.invoke({
        "filing_text": state["filing_text"]
    })
    return {"extracted_signals": response}  # Only return changed keys

def score_dimensions(state: OrgAIRState) -> dict:
    """Score the 7 AI readiness dimensions."""
    scores = scoring_chain.invoke({
        "signals": state["extracted_signals"],
        "ticker": state["ticker"]
    })
    final = sum(scores.values()) / len(scores)
    return {
        "dimension_scores": [scores],
        "final_score": final,
        "needs_human_review": final < 50  # HITL threshold
    }
```

### 5.3 Conditional Edges — The Routing Logic

```python
def should_review_or_report(state: OrgAIRState) -> str:
    """Route based on score and retry count."""
    if state["error_message"]:
        if state["retry_count"] < 3:
            return "retry_analysis"
        else:
            return "handle_error"
    elif state["needs_human_review"]:
        return "human_review_node"
    else:
        return "generate_investment_report"

graph.add_conditional_edges(
    "score_dimensions",           # source node
    should_review_or_report,      # routing function
    {
        "retry_analysis": "extract_ai_signals",    # back to start (loop!)
        "human_review_node": "human_review_node",
        "generate_investment_report": "generate_investment_report",
        "handle_error": END
    }
)
```

### 5.4 Checkpointing & HITL

```python
from langgraph.checkpoint.memory import MemorySaver

# Add checkpointer for persistence + HITL
checkpointer = MemorySaver()
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review_node"]  # Pause here for human input
)

# Run until interrupt
config = {"configurable": {"thread_id": "review-AAPL-001"}}
result = app.invoke(initial_state, config=config)

# Human analyst reviews and approves
human_input = {"approved": True, "analyst_notes": "Scores look reasonable"}

# Resume from checkpoint
final_result = app.invoke(
    {"human_approval": human_input},
    config=config
)
```

### 5.5 Streaming

```python
# Stream events from every node
for event in app.stream(initial_state, config=config):
    for node_name, node_output in event.items():
        print(f"[{node_name}] → {node_output}")

# Stream tokens from LLM calls
async for chunk in app.astream_events(initial_state, version="v2"):
    if chunk["event"] == "on_chat_model_stream":
        print(chunk["data"]["chunk"].content, end="", flush=True)
```

---

## 6. PE-OrgAIR Integration Examples

### 6.1 LangChain — Single Filing Analysis Chain

```
SEC EDGAR API
      │
      ▼
Document Chunker (750 words, 50 overlap)
      │
      ▼
RAG Retriever (top 5 relevant chunks)
      │
      ▼
ChatPromptTemplate (system: PE analyst, human: analyze {company})
      │
      ▼
ChatOpenAI / ChatAnthropic (via LiteLLM)
      │
      ▼
Pydantic Output Parser → AIReadinessScores
      │
      ▼
PostgreSQL (store scores) + Redis (cache results)
```

### 6.2 LangGraph — Multi-Agent Orchestration Workflow

```
                    START
                      │
              ┌───────▼────────┐
              │ fetch_filings  │ ◄─── SEC EDGAR Tool
              │ (Tool-calling) │ ◄─── S3/Snowflake Tool
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  extract_      │ ◄─── LLM: extract AI signals
              │  ai_signals    │      from 10-K filings
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  score_7_      │ ◄─── LLM: score each dimension
              │  dimensions    │      (Data Infra, Talent, etc.)
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │   calculate_   │
              │  final_score   │
              └───────┬────────┘
                      │
           ┌──────────┴──────────┐
           │ needs_review? (HITL) │
           └──────┬──────────────┘
         YES      │         NO
          ▼       │          ▼
   human_review   │    generate_report ──► Airflow DAG trigger
   (interrupt)    │          │
          │       │          │
          └───────┴──────────┘
                  │
                 END
```

### 6.3 FastAPI + LangGraph Streaming Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/analyze/{ticker}/stream")
async def stream_analysis(ticker: str):
    """Stream AI readiness analysis results in real-time."""
    
    async def event_generator():
        async for event in orgair_graph.astream_events(
            {"ticker": ticker},
            version="v2"
        ):
            if event["event"] == "on_chain_end":
                yield f"data: {json.dumps(event['data'])}\n\n"
            elif event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                yield f"data: {json.dumps({'token': token})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

---

## 7. Architecture Patterns

### Pattern 1: Linear Chain (LangChain)

```
Use when: Single analysis pass, no branching needed

Input → Validate → Extract → Score → Format → Output
  ↑_______________________________________________|
                    No feedback loops
```

### Pattern 2: Stateful Agent with Tools (LangGraph)

```
Use when: Need to call external APIs, retry on failure

START → Agent ──► Tool Call ──► Agent ──► END
              ↑__________________|
              (tool result fed back)
```

### Pattern 3: Multi-Agent Supervisor (LangGraph)

```
Use when: Different specialized agents for different tasks

              ┌─────────────┐
              │  Supervisor │
              └──┬──────┬───┘
                 │      │
       ┌─────────▼─┐  ┌─▼──────────┐
       │ SEC Agent │  │Scoring Agent│
       │(filings)  │  │(7 dims)    │
       └───────────┘  └────────────┘
```

### Pattern 4: Human-in-the-Loop (LangGraph)

```
Use when: PE analysts must approve investment recommendations

Analyze → Score → [HITL checkpoint] → Approve/Reject → Report
               ↑                             │
               └─────── Request changes ─────┘
```

---

## 8. Quick Reference

### LangChain Cheatsheet

```python
# Chain composition
chain = prompt | llm | parser

# Parallel
from langchain_core.runnables import RunnableParallel
parallel = RunnableParallel({"a": chain_a, "b": chain_b})

# Structured output
llm.with_structured_output(PydanticModel)

# Stream tokens
for chunk in chain.stream(input):
    print(chunk, end="")

# Async
result = await chain.ainvoke(input)
```

### LangGraph Cheatsheet

```python
# Build graph
graph = StateGraph(MyState)
graph.add_node("node_name", function)
graph.set_entry_point("first_node")
graph.add_edge("node_a", "node_b")
graph.add_conditional_edges("node", router_fn, {"route": "target_node"})
graph.add_edge("last_node", END)

# Compile
app = graph.compile()
app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["node"])

# Run
result = app.invoke(initial_state)
for event in app.stream(initial_state):
    print(event)

# Resume after interrupt
app.invoke(updated_state, config={"configurable": {"thread_id": "xyz"}})
```

### The 7 PE-OrgAIR Dimensions

| # | Dimension | Weight | Key Signals in Filings |
|---|---|---|---|
| 1 | Data Infrastructure | 22% | Data lakes, pipelines, quality metrics |
| 2 | Technology Stack | 18% | Cloud platforms, ML frameworks, APIs |
| 3 | Talent Capability | 15% | AI/ML hires, PhDs, training programs |
| 4 | Leadership Vision | 14% | AI strategy mentions, executive buy-in |
| 5 | AI Governance | 12% | Risk frameworks, ethics policies |
| 6 | Use Case Portfolio | 12% | Deployed AI products, ROI evidence |
| 7 | Culture & Change | 7% | Change management, experimentation culture |

---

## Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
langchain = "^0.3.0"
langchain-openai = "^0.2.0"
langchain-anthropic = "^0.2.0"
langgraph = "^0.2.0"
langgraph-checkpoint = "^2.0.0"
litellm = "^1.50.0"          # Multi-provider LLM routing
pydantic = "^2.0.0"
fastapi = "^0.115.0"
```

---

## Further Reading

- [LangChain Docs](https://python.langchain.com/docs/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph How-To: HITL](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
- [LCEL Conceptual Guide](https://python.langchain.com/docs/concepts/lcel/)

---

*SpringBigData — PE-OrgAIR Platform | Lab 5: LLM-Powered Intelligence Layer*