"""
PE-OrgAIR AI Readiness Analyzer — Streamlit Frontend
=====================================================
Dual Engine: LangChain (linear) + LangGraph (stateful graph)
Run:  streamlit run app.py
"""

import streamlit as st
import requests
import json
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="PE-OrgAIR: AI Readiness Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .engine-langchain { border-left: 4px solid #3b82f6; padding-left: 12px; }
    .engine-langgraph { border-left: 4px solid #10b981; padding-left: 12px; }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("⚙️ Configuration")

    # ENGINE TOGGLE — the key new feature
    st.subheader("🔀 Engine Selection")
    engine = st.radio(
        "Choose engine",
        ["LangChain", "LangGraph"],
        help=(
            "**LangChain:** Linear agent — same flow every time (A → B → C → done).\n\n"
            "**LangGraph:** Stateful graph — conditional edges, retry loops when "
            "confidence is low, memory across conversations."
        ),
    )
    engine_key = engine.lower().replace(" ", "")  # "langchain" or "langgraph"

    if engine == "LangChain":
        st.info("🔵 **Linear flow** — Agent calls tools in sequence, same path every run.")
    else:
        st.success(
            "🟢 **Stateful graph** — Conditional edges decide the path. "
            "If scoring confidence < 0.7, the graph loops back to gather more evidence."
        )

    st.divider()

    analysis_type = st.radio(
        "Analysis Type",
        ["Full Analysis", "Quick Assessment", "Risk Only"],
    )
    type_mapping = {
        "Full Analysis": "full",
        "Quick Assessment": "quick",
        "Risk Only": "risk_only",
    }

    st.divider()

    # Health check
    st.subheader("🔌 Backend Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("✅ Connected")
        engines = health.get("engines", [])
        st.caption(f"Engines: {', '.join(engines)}")
        st.caption(f"SerpAPI: {'✅' if health.get('serpapi_configured') else '❌'}")
    except Exception:
        st.error("❌ Backend offline")
        st.code("uvicorn main:app --reload", language="bash")

    st.divider()
    use_sample = st.checkbox("Use sample filing data", value=False)

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("🤖 PE-OrgAIR: AI Readiness Analysis Agent")

# Show which engine is active
if engine == "LangChain":
    st.markdown("*Using **🔵 LangChain** engine — Linear agent with tool calling*")
else:
    st.markdown("*Using **🟢 LangGraph** engine — Stateful graph with conditional edges & retry loops*")

st.divider()

# --- Input Section ---
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📝 Company Information")
    company_name = st.text_input("Company Name", value="TechCorp Industries")
with col2:
    st.subheader("📊 Analysis Settings")
    ca, cb, cc = st.columns(3)
    with ca:
        st.metric("Engine", engine_key)
    with cb:
        st.metric("Type", type_mapping[analysis_type])
    with cc:
        st.metric("Max Iter", "7")

st.divider()
st.subheader("📄 SEC Filing Text")

SAMPLE_TEXT = """The Company has made significant investments in artificial intelligence and machine learning \
capabilities to drive operational efficiency and enhance customer experiences. In fiscal 2024, we invested \
$75 million in AI infrastructure, talent acquisition, and research initiatives.

Key AI initiatives include:
- Establishment of an AI Center of Excellence with 150+ specialized engineers and data scientists
- Development of proprietary machine learning models for demand forecasting with 92% accuracy
- Implementation of generative AI for customer service automation, targeting 40% improvement in response times
- Supply chain optimization using advanced analytics and predictive modeling
- Strategic partnerships with leading AI research institutions including Stanford HAI
- Deployment of computer vision systems across 200+ warehouse facilities

Our AI governance framework includes a dedicated AI Ethics Board established in 2023, model risk management \
policies aligned with NIST AI RMF, and quarterly bias audits on all production models.

We face competitive risks from both established technology firms and well-funded startups entering our market. \
Regulatory scrutiny around AI ethics and transparency may impact our deployment timelines. Talent competition \
for AI specialists remains intense, with salary pressures increasing 15% year-over-year.

We expect these investments to generate approximately $200 million in cumulative cost savings over three years \
and establish competitive advantages in key market segments. Our Chief Data Officer reports directly to the CEO \
and oversees all AI strategy and governance."""

if use_sample:
    filing_text = st.text_area("Filing Content", value=SAMPLE_TEXT, height=200)
else:
    filing_text = st.text_area(
        "Filing Content",
        placeholder="Paste the relevant section from the SEC filing here...",
        height=200,
    )

# ============================================================================
# ANALYSIS EXECUTION
# ============================================================================
col1, col2, _ = st.columns([1, 1, 1])
with col1:
    analyze_button = st.button("🚀 Analyze AI Readiness", use_container_width=True, type="primary")
with col2:
    if st.button("🔄 Clear Results", use_container_width=True):
        if "analysis_result" in st.session_state:
            del st.session_state["analysis_result"]
        st.rerun()

if analyze_button and filing_text.strip():
    st.divider()

    engine_emoji = "🔵" if engine_key == "langchain" else "🟢"
    agent_status = st.status(f"{engine_emoji} Running {engine} agent...", state="running")

    with agent_status:
        st.write(f"🔗 Connecting to `{API_URL}/analyze` with **{engine}** engine")

        try:
            payload = {
                "company_name": company_name,
                "filing_text": filing_text,
                "analysis_type": type_mapping[analysis_type],
                "engine": engine_key,
            }

            response = requests.post(
                f"{API_URL}/analyze", json=payload, stream=True, timeout=180,
            )
            response.raise_for_status()
            st.write("✅ Connected. Processing...")

            final_result = None
            step_count = 0

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                evt_type = data.get("type")

                if evt_type == "status":
                    step_count += 1
                    msg = data.get("message", "")

                    # Highlight retry loops in LangGraph
                    if "gather_more_evidence" in msg or "Re-scoring" in msg:
                        st.write(f"**Step {step_count}:** 🔄 {msg}")
                    else:
                        st.write(f"**Step {step_count}:** {msg}")

                elif evt_type == "final":
                    final_result = data.get("result")
                    st.success("✅ Analysis complete!")

                elif evt_type == "error":
                    st.error(f"❌ {data.get('message', 'Unknown error')}")
                    break

            st.session_state["analysis_result"] = final_result
            agent_status.update(
                label=f"✅ {engine} Analysis Complete" if final_result else "⚠️ No result",
                state="complete" if final_result else "error",
            )

        except requests.exceptions.ConnectionError:
            agent_status.update(label="❌ Connection Failed", state="error")
            st.error(f"Cannot connect to backend at `{API_URL}`. Is it running?")
        except requests.exceptions.Timeout:
            agent_status.update(label="❌ Timeout", state="error")
            st.error("Request timed out after 180 seconds.")
        except Exception as e:
            agent_status.update(label="❌ Error", state="error")
            st.error(f"Error: {str(e)}")

# ============================================================================
# RESULTS DISPLAY
# ============================================================================
if "analysis_result" in st.session_state and st.session_state["analysis_result"]:
    final_data = st.session_state["analysis_result"]
    used_engine = final_data.get("engine", "unknown")

    st.divider()

    # Engine badge
    if used_engine == "langgraph":
        st.markdown("**Results from 🟢 LangGraph engine** (stateful graph with conditional edges)")
    else:
        st.markdown("**Results from 🔵 LangChain engine** (linear agent)")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "🔍 Details", "🔀 Graph Info", "💾 Raw JSON"])

    # ── Tab 1: Score Overview ──
    with tab1:
        score = final_data.get("readiness_score", 0)
        initiatives = final_data.get("ai_initiatives", [])
        risks = final_data.get("risks", [])

        c1, c2, c3 = st.columns(3)
        with c1:
            if score >= 75:
                label = "🟢 Leading"
            elif score >= 50:
                label = "🟡 Competitive"
            elif score >= 25:
                label = "🟠 Developing"
            else:
                label = "🔴 Nascent"
            st.metric("AI Readiness Score", f"{score:.0f}/100", delta=label)
        with c2:
            st.metric("AI Initiatives", len(initiatives))
        with c3:
            st.metric("Risk Factors", len(risks))

        # Dimension scores
        dim_scores = final_data.get("dimension_scores", {})
        if dim_scores:
            st.divider()
            st.subheader("📈 Dimension Scores (PE-OrgAIR 7 Dimensions)")
            cols = st.columns(min(len(dim_scores), 7))
            for i, (dim, val) in enumerate(dim_scores.items()):
                with cols[i % len(cols)]:
                    display_name = dim.replace("_", " ").title()
                    try:
                        v = float(val)
                        color = "🟢" if v >= 70 else "🟡" if v >= 50 else "🔴"
                        st.metric(display_name, f"{v:.0f}", delta=color)
                    except (ValueError, TypeError):
                        st.metric(display_name, str(val))

        st.divider()
        st.subheader("📋 Executive Summary")
        st.write(final_data.get("summary", "No summary available."))

    # ── Tab 2: Details ──
    with tab2:
        if initiatives:
            st.subheader("🚀 AI Initiatives")
            for i, init in enumerate(initiatives, 1):
                title = init.get("title", init.get("name", f"Initiative {i}"))
                with st.expander(f"**{i}. {title}**"):
                    st.write(f"**Description:** {init.get('description', 'N/A')}")
                    ca, cb = st.columns(2)
                    with ca:
                        st.write(f"**Investment:** {init.get('investment_level', init.get('investment', 'N/A'))}")
                    with cb:
                        st.write(f"**Timeline:** {init.get('timeline', 'N/A')}")

        st.divider()

        if risks:
            st.subheader("⚠️ Risk Assessment")
            for i, risk in enumerate(risks, 1):
                severity = str(risk.get("severity", "Unknown")).lower()
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
                cat = risk.get("category", "General")
                desc = risk.get("description", "N/A")
                with st.expander(f"{icon} **{cat}**: {desc}"):
                    st.write(f"**Severity:** {severity.title()}")
                    if risk.get("mitigation"):
                        st.write(f"**Mitigation:** {risk['mitigation']}")

    # ── Tab 3: Graph Info (LangGraph specific) ──
    with tab3:
        st.subheader("🔀 Engine Comparison")

        if used_engine == "langgraph":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Engine", "🟢 LangGraph")
            with c2:
                st.metric("Scoring Iterations", final_data.get("scoring_iterations", 1))
            with c3:
                st.metric("Confidence", f"{final_data.get('confidence', 0):.2f}")

            if final_data.get("scoring_iterations", 1) > 1:
                st.info(
                    f"🔄 **Retry loop activated!** The graph scored the company "
                    f"{final_data['scoring_iterations']} times because initial confidence "
                    f"was below 0.7. It gathered additional evidence and re-scored."
                )

            st.divider()
            st.subheader("Graph Flow Executed")
            st.code("""
START
  │
  ▼
search_news ──────────────────────────┐
  │                                    │
  ▼                                    │
extract_initiatives                    │
  │                                    │
  ▼                                    │
extract_risks                          │
  │                                    │
  ▼                                    │
score_company ◄──────────────┐         │
  │                           │         │
  ▼                           │         │
[confidence check]            │         │
  │              │            │         │
  │ >= 0.7       │ < 0.7      │         │
  ▼              ▼            │         │
build_final   gather_more ────┘         │
_answer       _evidence                 │
  │                                    │
  ▼                                    │
 END                                   │
            """, language="text")

        else:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Engine", "🔵 LangChain")
            with c2:
                st.metric("Flow", "Linear")

            st.info("🔵 **Linear flow** — The agent called tools in sequence with no branching or retries.")

            st.code("""
START → LLM thinks → call tool → LLM thinks → call tool → ... → final_answer → END

Always the same path. No loops. No decisions.
            """, language="text")

        st.divider()
        st.subheader("When to Use Which?")
        st.markdown("""
| Scenario | Best Engine |
|----------|------------|
| Quick one-off analysis | 🔵 LangChain |
| High-confidence scoring needed | 🟢 LangGraph (retries until confident) |
| Simple prompt → response | 🔵 LangChain |
| Multi-turn conversation | 🟢 LangGraph (checkpointing) |
| Production batch scoring | 🟢 LangGraph (deterministic graph) |
| Debugging / understanding flow | 🟢 LangGraph (explicit nodes) |
        """)

    # ── Tab 4: Raw JSON ──
    with tab4:
        st.json(final_data)

elif analyze_button:
    st.warning("⚠️ Please enter filing text before analyzing.")
else:
    if "analysis_result" not in st.session_state:
        st.info("""
        **How to use:**
        1. Select engine in the sidebar (**LangChain** or **LangGraph**)
        2. Enter company name and paste SEC filing text
        3. Click **Analyze AI Readiness**
        4. Compare results between engines!
        
        **Try this:** Run the same filing with both engines and compare
        the results in the **Graph Info** tab to see the difference.
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown(
    f"""<div style="text-align:center; color:gray; font-size:12px;">
    PE-OrgAIR Capstone | LangChain + LangGraph + Claude + FastAPI + Streamlit | Backend: {API_URL}
    </div>""",
    unsafe_allow_html=True,
)