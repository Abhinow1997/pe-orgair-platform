import os
from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_agent

llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

print("TOOLS + AGENTS: Simple Example with Tool Call Tracking")
print("=" * 70)

# ============================================================================
# UNDERSTANDING TOOLS + AGENTS
# ============================================================================
# TOOL: A function that the LLM can call to do something
#   - Examples: fetch data, calculate scores, send emails, query databases
#
# AGENT: An LLM that decides WHICH tools to use and WHEN to use them
#   - The LLM thinks: "I need to solve this problem, what tools do I need?"
#   - It calls the right tools in the right order
#   - It keeps going until the problem is solved
# ============================================================================

# ============================================================================
# STEP 1: Define TOOLS (functions the LLM can call)
# ============================================================================

# Tool 1: Get company AI readiness score from database
@tool
def get_ai_readiness_score(company_name: str) -> str:
    """
    Get the AI readiness score for a company.
    Returns a score from 0-100 and assessment summary.
    """
    # Simulating a database lookup
    scores = {
        "Walmart": "75 - Strong AI foundation, good data infrastructure",
        "Target": "62 - Moderate AI adoption, legacy systems",
        "Amazon": "92 - Leading AI implementation across all operations",
    }
    result = scores.get(company_name, f"No data found for {company_name}")
    print(f"  📊 [TOOL CALLED] get_ai_readiness_score('{company_name}') → {result}")
    return result


# Tool 2: Check if a company is a good acquisition target
@tool
def check_acquisition_fit(ai_score: int, industry: str) -> str:
    """
    Determine if a company is a good acquisition target based on AI readiness.
    Takes an AI score (0-100) and industry type.
    """
    if ai_score >= 80:
        result = f"EXCELLENT FIT: High AI maturity. {industry} companies with this score are prime acquisition targets."
    elif ai_score >= 60:
        result = f"GOOD FIT: Moderate AI readiness. {industry} companies need some integration work but have strong foundation."
    else:
        result = f"RISKY: Low AI readiness. {industry} companies will require significant AI transformation post-acquisition."
    
    print(f"  🎯 [TOOL CALLED] check_acquisition_fit(ai_score={ai_score}, industry='{industry}') → {result}")
    return result


# Tool 3: Generate integration plan
@tool
def generate_integration_plan(company_name: str, ai_score: int) -> str:
    """
    Generate a 90-day integration plan based on company AI readiness.
    """
    if ai_score >= 80:
        result = f"{company_name} Integration Plan (90 days):\n  1. Consolidate AI teams\n  2. Align ML models with PE platform\n  3. Scale successful initiatives"
    else:
        result = f"{company_name} Integration Plan (90 days):\n  1. Audit legacy systems\n  2. Build data infrastructure\n  3. Hire/train AI talent"
    
    print(f"  📋 [TOOL CALLED] generate_integration_plan('{company_name}', ai_score={ai_score})")
    return result


# ============================================================================
# STEP 2: Create an AGENT that uses these tools
# ============================================================================

tools = [get_ai_readiness_score, check_acquisition_fit, generate_integration_plan]

# Create the agent using LangGraph
agent = create_agent(llm, tools)

# ============================================================================
# STEP 3: Give the agent a task
# ============================================================================

print("\n[AGENT TASK] Evaluate whether Walmart is a good acquisition target\n")
print("=" * 70)

task = """
You are a Private Equity analyst. Your job is to:
1. Get Walmart's AI readiness score
2. Check if it's a good acquisition fit in retail
3. Generate an integration plan if it's a good target

Give me your final recommendation.
"""

print("\n[AGENT WORKFLOW - Tool Calls in Order]\n")

# The agent will:
#   - Read the task
#   - Decide which tools to call
#   - Call them in the right order
#   - Use the results to make a decision
result = agent.invoke({"messages": [{"role": "user", "content": task}]})

print("\n" + "=" * 70)
print("\n[FINAL AGENT RECOMMENDATION]\n")
print(result["messages"][-1].content)