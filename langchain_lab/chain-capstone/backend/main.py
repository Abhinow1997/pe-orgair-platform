"""
PE-OrgAIR AI Readiness Analysis Agent — FastAPI Backend
========================================================
Dual Engine: LangChain (linear agent) + LangGraph (stateful graph)

Run:   uvicorn main:app --reload
Deps:  pip install fastapi uvicorn langchain langchain-anthropic langchain-core
       pip install langgraph python-dotenv pydantic google-search-results
"""

import asyncio
import json
import os
import uuid
from typing import Optional, TypedDict, Annotated
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ============================================================================
# CONFIGURATION
# ============================================================================
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
MAX_ITERATIONS = 7

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class CompanyAnalysisRequest(BaseModel):
    company_name: str
    filing_text: str = Field(..., description="10-K or 10-Q filing excerpt")
    analysis_type: str = Field(default="full", description="full, quick, risk_only")
    engine: str = Field(default="langchain", description="langchain or langgraph")


# ============================================================================
# LLM SETUP
# ============================================================================
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.0,
    api_key=ANTHROPIC_API_KEY,
    max_tokens=4096,
    timeout=60.0,
)

system_prompt = """You are an expert AI Readiness analyst for Private Equity firms evaluating portfolio companies using the PE-OrgAIR framework.

You have access to these tools — use them in this order:

1. **search_company_news** — Search the web for recent AI-related news about the company (use this FIRST to get current context beyond the filing)
2. **extract_ai_initiatives** — Extract AI initiatives from the filing text + web research
3. **extract_risk_factors** — Identify AI-related risk factors
4. **calculate_readiness_score** — Compute the V^R Idiosyncratic Readiness score (0-100)
5. **final_answer** — Provide the structured final analysis

IMPORTANT RULES:
- Always start with search_company_news to get current context
- Use ALL tools in sequence before calling final_answer
- Be thorough: cite specific dollar amounts, team sizes, and timelines
- The readiness score must follow the PE-OrgAIR rubric levels:
  Level 5 (80-100): Industry-leading AI capabilities
  Level 4 (60-79): Strong, established AI programs  
  Level 3 (40-59): Developing, modernizing
  Level 2 (20-39): Early stage, limited
  Level 1 (0-19): No meaningful AI capability
"""


# ============================================================================
# TOOLS DEFINITION (shared by both engines)
# ============================================================================
@tool
def search_company_news(query: str) -> str:
    """
    Search the web for recent AI-related news about a company.
    Use this to supplement SEC filing data with current information.
    Pass a search query like 'NVIDIA AI initiatives 2024 2025'.
    """
    if not SERPAPI_API_KEY:
        return json.dumps({
            "note": "SerpAPI key not configured. Using filing data only.",
            "results": []
        })

    try:
        from serpapi import GoogleSearch

        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": 5,
            "tbm": "nws",
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        news_results = []
        for item in results.get("news_results", [])[:5]:
            news_results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", {}).get("name", ""),
                "date": item.get("date", ""),
                "link": item.get("link", ""),
            })

        if not news_results:
            for item in results.get("organic_results", [])[:5]:
                news_results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("displayed_link", ""),
                    "link": item.get("link", ""),
                })

        return json.dumps({
            "query": query,
            "result_count": len(news_results),
            "results": news_results
        }, indent=2)

    except ImportError:
        return json.dumps({"error": "serpapi not installed", "results": []})
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})


@tool
def extract_ai_initiatives(filing_text: str, company: str, web_context: str = "") -> str:
    """
    Extract AI initiatives from SEC filing text and web research.
    Identifies: AI projects, investments, partnerships, team sizes, timelines.
    """
    extraction_llm = ChatAnthropic(
        model="claude-sonnet-4-20250514", temperature=0.0,
        api_key=ANTHROPIC_API_KEY, max_tokens=1500,
    )
    response = extraction_llm.invoke(
        f"""Extract AI initiatives from this text. Return ONLY valid JSON, no markdown.

Text: {filing_text[:3000]}
Additional context: {web_context[:500] if web_context else 'None'}

Return JSON format:
{{"initiatives": [{{"title": "...", "description": "...", "investment_level": "High/Medium/Low", "timeline": "..."}}], "total_ai_budget_estimate": "..."}}"""
    )
    return response.content


@tool
def extract_risk_factors(filing_text: str, company: str) -> str:
    """
    Extract AI-related risk factors from SEC filing.
    Categories: Regulatory, Competitive, Technical, Talent, Ethical.
    """
    extraction_llm = ChatAnthropic(
        model="claude-sonnet-4-20250514", temperature=0.0,
        api_key=ANTHROPIC_API_KEY, max_tokens=1500,
    )
    response = extraction_llm.invoke(
        f"""Extract AI-related risk factors from this filing text. Return ONLY valid JSON, no markdown.

Company: {company}
Text: {filing_text[:3000]}

Return JSON format:
{{"risks": [{{"category": "Regulatory|Competitive|Technical|Talent|Ethical", "description": "...", "severity": "High|Medium|Low", "mitigation": "..."}}]}}"""
    )
    return response.content


@tool
def calculate_readiness_score(
    initiatives_summary: str,
    risks_summary: str,
    company: str,
    sector: str = "technology"
) -> str:
    """
    Calculate composite AI Readiness Score (V^R) using PE-OrgAIR rubric.
    Score 0-100 across 7 dimensions with weighted average.
    """
    scoring_llm = ChatAnthropic(
        model="claude-sonnet-4-20250514", temperature=0.0,
        api_key=ANTHROPIC_API_KEY, max_tokens=1500,
    )
    response = scoring_llm.invoke(
        f"""You are a PE-OrgAIR scoring engine. Score {company} ({sector} sector).

Initiatives: {initiatives_summary[:2000]}
Risks: {risks_summary[:1000]}

Score each dimension 0-100, then compute weighted average:
- Data Infrastructure (18%)
- AI Governance (15%)
- Technology Stack (15%)
- Talent (17%)
- Leadership (13%)
- Use Case Portfolio (12%)
- Culture (10%)

Return ONLY valid JSON, no markdown:
{{"dimensions": {{"data_infrastructure": 0, "ai_governance": 0, "technology_stack": 0, "talent": 0, "leadership": 0, "use_case_portfolio": 0, "culture": 0}}, "weighted_score": 0, "category": "Leading|Competitive|Developing|Nascent", "confidence": 0.0, "rationale": "..."}}"""
    )
    return response.content


@tool
def final_answer(
    summary: str,
    ai_initiatives: list[dict],
    risks: list[dict],
    readiness_score: float,
    dimension_scores: dict = None,
    web_sources: list[str] = None
) -> str:
    """
    Provide the final structured answer. Call this LAST after all other tools.
    """
    result = {
        "summary": summary,
        "ai_initiatives": ai_initiatives,
        "risks": risks,
        "readiness_score": readiness_score,
        "dimension_scores": dimension_scores or {},
        "web_sources": web_sources or [],
        "timestamp": datetime.now().isoformat(),
    }
    return json.dumps(result)


# All tools list and map
all_tools = [
    search_company_news,
    extract_ai_initiatives,
    extract_risk_factors,
    calculate_readiness_score,
    final_answer,
]
tool_map = {t.name: t for t in all_tools}


# ============================================================================
# ENGINE 1: LANGCHAIN — Simple ReAct-style loop (from before)
# ============================================================================
async def run_langchain_agent(input_text: str, queue: asyncio.Queue):
    """
    LangChain engine: Manual tool-calling loop.

    How it works:
      - Send messages to LLM with tools bound
      - If LLM returns tool_calls → execute them, append results, loop
      - If LLM returns plain text → done
      - Safety: stop after MAX_ITERATIONS

    This is a LINEAR flow — the agent always follows the same pattern:
      Think → Call tool → Think → Call tool → ... → Final answer
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text},
    ]

    llm_with_tools = llm.bind_tools(all_tools, tool_choice="auto")
    final_result = None

    for iteration in range(MAX_ITERATIONS):
        await queue.put({
            "type": "status",
            "message": f"[LangChain] Step {iteration + 1}/{MAX_ITERATIONS} — Thinking..."
        })

        response: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            await queue.put({
                "type": "status",
                "message": "[LangChain] Agent finished (no more tool calls)."
            })
            if not final_result:
                final_result = {
                    "summary": response.content,
                    "ai_initiatives": [], "risks": [],
                    "readiness_score": 0, "dimension_scores": {},
                    "web_sources": [],
                    "engine": "langchain",
                    "timestamp": datetime.now().isoformat(),
                }
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            await queue.put({
                "type": "status",
                "message": f"[LangChain] 🔧 Calling: {tool_name}"
            })

            try:
                tool_fn = tool_map[tool_name]
                result = tool_fn.invoke(tool_args)

                if tool_name == "final_answer":
                    try:
                        final_result = json.loads(result)
                        final_result["engine"] = "langchain"
                    except (json.JSONDecodeError, TypeError):
                        final_result = {
                            "summary": str(result),
                            "ai_initiatives": [], "risks": [],
                            "readiness_score": 0, "engine": "langchain",
                            "timestamp": datetime.now().isoformat(),
                        }
                    await queue.put({"type": "status", "message": "[LangChain] ✅ Final answer generated!"})

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                await queue.put({"type": "status", "message": f"[LangChain] ✅ {tool_name} completed"})

            except Exception as e:
                messages.append(ToolMessage(content=json.dumps({"error": str(e)}), tool_call_id=tool_id))
                await queue.put({"type": "status", "message": f"[LangChain] ⚠️ {tool_name} failed: {e}"})

        if final_result:
            break

    if not final_result:
        final_result = {
            "summary": "Max iterations reached.", "ai_initiatives": [], "risks": [],
            "readiness_score": 0, "engine": "langchain", "timestamp": datetime.now().isoformat(),
        }

    await queue.put({"type": "final", "result": final_result})
    return final_result


# ============================================================================
# ENGINE 2: LANGGRAPH — Stateful Graph with Conditional Edges
# ============================================================================

# --- Step 1: Define State ---
class AgentState(TypedDict):
    """
    The shared whiteboard for the LangGraph agent.

    Every node reads from this state and writes updates to it.
    'messages' uses add_messages reducer → new messages APPEND (don't replace).
    All other fields REPLACE on update.
    """
    messages: Annotated[list, add_messages]
    company: str
    filing_text: str
    analysis_type: str
    web_context: str
    initiatives: str
    risks: str
    score_data: str
    final_result: dict
    confidence: float
    iteration: int
    status_queue: object  # asyncio.Queue for streaming status


# --- Step 2: Define Nodes ---

async def search_news_node(state: AgentState) -> dict:
    """
    Node 1: Search for recent company news.

    Reads: company, filing_text
    Writes: web_context, messages
    """
    q = state.get("status_queue")
    if q:
        await q.put({"type": "status", "message": "[LangGraph] 🔍 Node: search_news — Searching web..."})

    query = f"{state['company']} AI initiatives artificial intelligence 2024 2025"
    result = search_company_news.invoke({"query": query})

    if q:
        await q.put({"type": "status", "message": "[LangGraph] ✅ search_news completed"})

    return {
        "web_context": result,
        "messages": [AIMessage(content=f"Web search completed. Found context for {state['company']}.")],
    }


async def extract_initiatives_node(state: AgentState) -> dict:
    """
    Node 2: Extract AI initiatives from filing + web context.

    Reads: filing_text, company, web_context
    Writes: initiatives, messages
    """
    q = state.get("status_queue")
    if q:
        await q.put({"type": "status", "message": "[LangGraph] 📊 Node: extract_initiatives — Analyzing filing..."})

    result = extract_ai_initiatives.invoke({
        "filing_text": state["filing_text"],
        "company": state["company"],
        "web_context": state.get("web_context", ""),
    })

    if q:
        await q.put({"type": "status", "message": "[LangGraph] ✅ extract_initiatives completed"})

    return {
        "initiatives": result,
        "messages": [AIMessage(content="AI initiatives extracted from filing.")],
    }


async def extract_risks_node(state: AgentState) -> dict:
    """
    Node 3: Extract risk factors.

    Reads: filing_text, company
    Writes: risks, messages
    """
    q = state.get("status_queue")
    if q:
        await q.put({"type": "status", "message": "[LangGraph] ⚠️ Node: extract_risks — Identifying risks..."})

    result = extract_risk_factors.invoke({
        "filing_text": state["filing_text"],
        "company": state["company"],
    })

    if q:
        await q.put({"type": "status", "message": "[LangGraph] ✅ extract_risks completed"})

    return {
        "risks": result,
        "messages": [AIMessage(content="Risk factors extracted.")],
    }


async def score_company_node(state: AgentState) -> dict:
    """
    Node 4: Calculate readiness score across 7 dimensions.

    Reads: initiatives, risks, company
    Writes: score_data, confidence, iteration, messages
    """
    q = state.get("status_queue")
    iteration = state.get("iteration", 0) + 1

    if q:
        await q.put({
            "type": "status",
            "message": f"[LangGraph] 🧮 Node: score_company (attempt {iteration}) — Calculating V^R..."
        })

    result = calculate_readiness_score.invoke({
        "initiatives_summary": state.get("initiatives", ""),
        "risks_summary": state.get("risks", ""),
        "company": state["company"],
    })

    # Parse confidence from result
    confidence = 0.5
    try:
        parsed = json.loads(result)
        confidence = parsed.get("confidence", 0.5)
    except (json.JSONDecodeError, TypeError):
        pass

    if q:
        await q.put({
            "type": "status",
            "message": f"[LangGraph] ✅ score_company completed (confidence: {confidence:.2f})"
        })

    return {
        "score_data": result,
        "confidence": confidence,
        "iteration": iteration,
        "messages": [AIMessage(content=f"Scoring complete. Confidence: {confidence:.2f}")],
    }


async def gather_more_evidence_node(state: AgentState) -> dict:
    """
    Node 5 (RETRY): Gather additional evidence when confidence is low.

    This node only runs if the conditional edge routes here.
    It searches for more specific information to improve scoring confidence.

    Reads: company, confidence
    Writes: web_context (appended), messages
    """
    q = state.get("status_queue")
    if q:
        await q.put({
            "type": "status",
            "message": f"[LangGraph] 🔄 Node: gather_more_evidence — Confidence {state['confidence']:.2f} too low, researching more..."
        })

    # Search for more specific information
    queries = [
        f"{state['company']} AI ML team size hiring data scientists",
        f"{state['company']} chief data officer AI governance board",
        f"{state['company']} AI production deployment ROI results",
    ]
    query = queries[min(state.get("iteration", 0), len(queries) - 1)]
    extra_context = search_company_news.invoke({"query": query})

    existing = state.get("web_context", "")
    combined = existing + "\n\n--- Additional Research ---\n" + extra_context

    if q:
        await q.put({"type": "status", "message": "[LangGraph] ✅ Additional evidence gathered. Re-scoring..."})

    return {
        "web_context": combined,
        "messages": [AIMessage(content="Gathered additional evidence for higher confidence scoring.")],
    }


async def build_final_answer_node(state: AgentState) -> dict:
    """
    Node 6: Build the final structured result.

    Reads: company, score_data, initiatives, risks, web_context
    Writes: final_result, messages
    """
    q = state.get("status_queue")
    if q:
        await q.put({"type": "status", "message": "[LangGraph] 📋 Node: build_final_answer — Packaging results..."})

    # Parse all the data
    initiatives_list = []
    try:
        parsed = json.loads(state.get("initiatives", "{}"))
        initiatives_list = parsed.get("initiatives", [])
    except (json.JSONDecodeError, TypeError):
        pass

    risks_list = []
    try:
        parsed = json.loads(state.get("risks", "{}"))
        risks_list = parsed.get("risks", [])
    except (json.JSONDecodeError, TypeError):
        pass

    score = 0.0
    dim_scores = {}
    category = "Unknown"
    rationale = ""
    try:
        parsed = json.loads(state.get("score_data", "{}"))
        score = parsed.get("weighted_score", 0)
        dim_scores = parsed.get("dimensions", {})
        category = parsed.get("category", "Unknown")
        rationale = parsed.get("rationale", "")
    except (json.JSONDecodeError, TypeError):
        pass

    # Build summary using LLM
    summary_llm = ChatAnthropic(
        model="claude-sonnet-4-20250514", temperature=0.0,
        api_key=ANTHROPIC_API_KEY, max_tokens=500,
    )
    summary_response = summary_llm.invoke(
        f"""Write a 3-4 sentence executive summary of {state['company']}'s AI readiness.
Score: {score}/100 ({category})
Key initiatives: {len(initiatives_list)} found
Key risks: {len(risks_list)} found
Rationale: {rationale}

Be concise and professional. This is for a PE investment committee."""
    )

    final = {
        "summary": summary_response.content,
        "ai_initiatives": initiatives_list,
        "risks": risks_list,
        "readiness_score": score,
        "dimension_scores": dim_scores,
        "category": category,
        "confidence": state.get("confidence", 0.0),
        "web_sources": [],
        "scoring_iterations": state.get("iteration", 1),
        "engine": "langgraph",
        "timestamp": datetime.now().isoformat(),
    }

    if q:
        await q.put({"type": "status", "message": "[LangGraph] ✅ Final answer built!"})

    return {
        "final_result": final,
        "messages": [AIMessage(content="Analysis complete.")],
    }


# --- Step 3: Conditional Edge — The Key Differentiator ---

def should_retry_or_finish(state: AgentState) -> str:
    """
    CONDITIONAL EDGE: Decide whether to gather more evidence or finalize.

    This is what makes LangGraph different from LangChain.
    The graph DECIDES at runtime which path to take:

      confidence >= 0.7  →  go to build_final_answer (we're confident)
      confidence < 0.7 AND iterations < 3  →  loop back to gather_more_evidence
      confidence < 0.7 AND iterations >= 3  →  go to build_final_answer (give up retrying)
    """
    confidence = state.get("confidence", 0.0)
    iteration = state.get("iteration", 0)

    if confidence >= 0.7:
        return "build_final_answer"
    elif iteration < 3:
        return "gather_more_evidence"      # ← LOOP BACK — impossible in LangChain!
    else:
        return "build_final_answer"


# --- Step 4: Build the Graph ---

def build_scoring_graph():
    """
    Construct the LangGraph scoring pipeline.

    Visual:
        START
          │
          ▼
        search_news
          │
          ▼
        extract_initiatives
          │
          ▼
        extract_risks
          │
          ▼
        score_company ◄──────────────────┐
          │                               │
          ▼                               │
        [should_retry_or_finish]          │
          │                    │          │
          │ confidence >= 0.7  │ < 0.7    │
          │ OR max retries     │          │
          ▼                    ▼          │
        build_final_answer   gather_more ─┘
          │                  evidence
          ▼
         END
    """
    builder = StateGraph(AgentState)

    # Add all nodes
    builder.add_node("search_news", search_news_node)
    builder.add_node("extract_initiatives", extract_initiatives_node)
    builder.add_node("extract_risks", extract_risks_node)
    builder.add_node("score_company", score_company_node)
    builder.add_node("gather_more_evidence", gather_more_evidence_node)
    builder.add_node("build_final_answer", build_final_answer_node)

    # Wire the linear flow
    builder.add_edge(START, "search_news")
    builder.add_edge("search_news", "extract_initiatives")
    builder.add_edge("extract_initiatives", "extract_risks")
    builder.add_edge("extract_risks", "score_company")

    # THE CONDITIONAL EDGE — this is the magic
    builder.add_conditional_edges(
        "score_company",
        should_retry_or_finish,
        {
            "gather_more_evidence": "gather_more_evidence",
            "build_final_answer": "build_final_answer",
        }
    )

    # Loop back: gather_more_evidence → score_company (RE-SCORE with new data)
    builder.add_edge("gather_more_evidence", "score_company")

    # Final answer → END
    builder.add_edge("build_final_answer", END)

    # Compile with checkpointer for memory across conversations
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Create the graph once at startup
scoring_graph = build_scoring_graph()


async def run_langgraph_agent(input_text: str, company: str, filing_text: str,
                               analysis_type: str, queue: asyncio.Queue):
    """
    LangGraph engine: Stateful graph with conditional edges and retry loops.

    Key differences from LangChain engine:
    1. STATE persists across all nodes (shared whiteboard)
    2. CONDITIONAL EDGES let the graph decide: retry or finish
    3. LOOPS allow re-scoring when confidence is low
    4. CHECKPOINTING enables multi-turn memory via thread_id
    """
    thread_id = str(uuid.uuid4())

    await queue.put({
        "type": "status",
        "message": f"[LangGraph] Initializing graph (thread: {thread_id[:8]}...)"
    })

    initial_state = {
        "messages": [HumanMessage(content=input_text)],
        "company": company,
        "filing_text": filing_text,
        "analysis_type": analysis_type,
        "web_context": "",
        "initiatives": "",
        "risks": "",
        "score_data": "",
        "final_result": {},
        "confidence": 0.0,
        "iteration": 0,
        "status_queue": queue,
    }

    config = {"configurable": {"thread_id": thread_id}}

    # Run the graph
    try:
        result = await scoring_graph.ainvoke(initial_state, config)
        final = result.get("final_result", {})

        if not final:
            final = {
                "summary": "Graph completed but no final result produced.",
                "ai_initiatives": [], "risks": [], "readiness_score": 0,
                "engine": "langgraph", "timestamp": datetime.now().isoformat(),
            }

        await queue.put({"type": "final", "result": final})
        return final

    except Exception as e:
        error_result = {
            "summary": f"LangGraph error: {str(e)}",
            "ai_initiatives": [], "risks": [], "readiness_score": 0,
            "engine": "langgraph", "timestamp": datetime.now().isoformat(),
        }
        await queue.put({"type": "final", "result": error_result})
        return error_result


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="PE-OrgAIR Analysis Agent",
    version="3.0.0",
    description="Dual Engine: LangChain (linear) + LangGraph (stateful graph)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================
@app.post("/analyze")
async def analyze_company(request: CompanyAnalysisRequest):
    """
    Analyze company AI readiness.

    Set engine="langchain" for linear agent or engine="langgraph" for stateful graph.
    """
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    engine = request.engine.lower()

    async def event_generator():
        input_text = (
            f"Analyze {request.company_name}'s AI readiness.\n\n"
            f"SEC Filing Text:\n{request.filing_text}\n\n"
            f"Analysis Type: {request.analysis_type}"
        )

        if engine == "langgraph":
            task = asyncio.create_task(
                run_langgraph_agent(
                    input_text=input_text,
                    company=request.company_name,
                    filing_text=request.filing_text,
                    analysis_type=request.analysis_type,
                    queue=queue,
                )
            )
        else:
            task = asyncio.create_task(
                run_langchain_agent(input_text=input_text, queue=queue)
            )

        yield f"data: {json.dumps({'type': 'status', 'message': f'Agent started (engine: {engine})', 'session_id': session_id})}\n\n"

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=180.0)

                if event["type"] == "final":
                    yield f"data: {json.dumps({'type': 'final', 'result': event['result']})}\n\n"
                    break
                elif event["type"] == "status":
                    yield f"data: {json.dumps({'type': 'status', 'message': event['message']})}\n\n"
                elif event["type"] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': event['message']})}\n\n"
                    break

            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Timed out after 180s'})}\n\n"
                break
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break

        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engines": ["langchain", "langgraph"],
        "serpapi_configured": SERPAPI_API_KEY is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/")
async def root():
    return {
        "name": "PE-OrgAIR Analysis Agent",
        "version": "3.0.0",
        "engines": {
            "langchain": "Linear ReAct agent — same flow every time",
            "langgraph": "Stateful graph — conditional edges, retry loops, memory",
        },
        "tools": [t.name for t in all_tools],
        "endpoints": ["/analyze", "/health"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)