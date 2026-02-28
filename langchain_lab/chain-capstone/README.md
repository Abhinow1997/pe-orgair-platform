# PE-OrgAIR: LangChain + LangGraph Lab

AI Readiness Analysis Agent for SEC filings, built with a dual-engine backend:

- `LangChain` for linear tool-calling workflows
- `LangGraph` for stateful graph execution with controllable flow

The app scores a company using the PE-OrgAIR framework (7 dimensions, weighted 0-100 score) and returns structured analysis.

## What Is LangChain?

LangChain is an application framework for LLM-powered workflows. It helps you combine:

- prompts
- models
- tools/functions
- structured outputs
- chains/agents

In simple terms, LangChain is best for pipeline-style AI flows where steps are mostly sequential.

## What Is LangGraph?

LangGraph is a graph runtime for agent workflows with explicit state and routing.
It is useful when you need:

- multi-step state tracking
- branching decisions
- loops/retries
- checkpointing and resumability

In simple terms, LangGraph is best for flowchart-style AI flows where the next step can depend on current state.

## How They Work Together In This Lab

The backend (`backend/main.py`) supports two engines through request field `engine`:

- `langchain`: linear agent-style execution
- `langgraph`: stateful graph execution

Both engines share the same core tools:

1. `search_company_news`
2. `extract_ai_initiatives`
3. `extract_risk_factors`
4. `calculate_readiness_score`
5. `final_answer`

High-level flow:

1. User sends company name + filing excerpt.
2. Agent gathers optional web context (SerpAPI).
3. Agent extracts initiatives and risks.
4. Agent calculates PE-OrgAIR weighted score.
5. Agent returns structured JSON response (summary, risks, initiatives, scores).

## Project Structure

```text
langchain_lab/chain-capstone/
|- backend/
|  |- main.py         # FastAPI app, LangChain + LangGraph engines
|- frontend/
|  |- app.py          # Streamlit UI
|- .env               # Local API keys (not committed)
|- README.md
```

## Prerequisites

- Python 3.10+
- Anthropic API key (required)
- SerpAPI key (optional, enables web/news context)

## Setup

1. Install dependencies:

```bash
pip install fastapi uvicorn langchain langchain-anthropic langchain-core langgraph python-dotenv pydantic streamlit requests google-search-results
```

2. Create `.env` in `langchain_lab/chain-capstone/`:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
SERPAPI_API_KEY=your-serpapi-key-here
```

If `SERPAPI_API_KEY` is missing, the app still runs and analyzes filing text only.

## Run the App

Start backend:

```bash
cd backend
uvicorn main:app --reload
```

Backend endpoints:

- API root: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Start frontend in another terminal:

```bash
cd frontend
streamlit run app.py
```

Frontend:

- `http://localhost:8501`

## Basic Working Details

### Request Model

The backend accepts:

- `company_name`
- `filing_text`
- `analysis_type` (`full`, `quick`, `risk_only`)
- `engine` (`langchain`, `langgraph`)

### State and Execution

- In `langchain` mode, the agent runs a mostly sequential tool workflow.
- In `langgraph` mode, a graph state object carries messages/results across nodes, enabling more controlled orchestration.

### Output

The final response includes:

- summary
- extracted initiatives
- extracted risks
- readiness score
- dimension-level scores
- web sources (if available)

## Troubleshooting

- `ANTHROPIC_API_KEY not found`: ensure `.env` exists and key is valid.
- Backend connection errors: confirm `uvicorn main:app --reload` is running on port `8000`.
- SerpAPI not used: set `SERPAPI_API_KEY`; otherwise search tool returns empty/notice response.
- Slow responses/timeouts: reduce filing text size and retry.

## Tech Stack

- LLM: Claude (Anthropic)
- Agent Framework: LangChain
- Graph Orchestration: LangGraph
- API: FastAPI + Uvicorn
- UI: Streamlit
- Search: SerpAPI (Google News / web context)

Built for SpringBigData (Spring 2026).
