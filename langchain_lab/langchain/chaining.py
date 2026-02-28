import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

print("EXTENDED EXAMPLE: Glassdoor Culture Analyzer with Chaining")
print("=" * 70)

# ============================================================================
# CHAIN 1: Analyze Culture from Reviews
# ============================================================================
culture_system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an organizational culture analyst for a Private Equity firm.
You analyze employee reviews to assess a company's culture and AI readiness."""
)

culture_user_prompt = HumanMessagePromptTemplate.from_template(
    """Analyze these {review_count} Glassdoor reviews for {company} ({ticker}).

=== EMPLOYEE REVIEWS ===
{reviews}

=== YOUR TASK ===
Score each culture dimension from 0 to 100:

INNOVATION_SCORE: [0-100]
DATA_DRIVEN_SCORE: [0-100]
AI_AWARENESS_SCORE: [0-100]
CHANGE_READINESS_SCORE: [0-100]
OVERALL_CULTURE_SCORE: [0-100]

POSITIVE_SIGNALS: [list key positive phrases]
NEGATIVE_SIGNALS: [list key negative phrases]
CONFIDENCE: [low/medium/high]""",
    input_variables=["company", "ticker", "review_count", "reviews"]
)

culture_prompt = ChatPromptTemplate.from_messages(
    [culture_system_prompt, culture_user_prompt]
)

chain_one = culture_prompt | llm | StrOutputParser()

# ============================================================================
# CHAIN 2: Generate Investment Recommendation Based on Culture Scores
# ============================================================================
investment_system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an investment analyst for a Private Equity firm evaluating 
companies for acquisition based on their organizational AI readiness."""
)

investment_user_prompt = HumanMessagePromptTemplate.from_template(
    """Based on the following culture analysis for {company} ({ticker}):

=== CULTURE ANALYSIS ===
{culture_analysis}

=== YOUR TASK ===
Provide a brief investment recommendation (3-4 sentences) that considers:
- AI readiness potential
- Change management capability
- Data-driven decision-making culture
- Risk factors for integration

Format:
RECOMMENDATION: [BUY / HOLD / PASS]
RATIONALE: [Your analysis]
KEY_RISKS: [Main concerns for PE integration]""",
    input_variables=["company", "ticker", "culture_analysis"]
)

investment_prompt = ChatPromptTemplate.from_messages(
    [investment_system_prompt, investment_user_prompt]
)

chain_two = (
    {
        "company": lambda x: x["company"],
        "ticker": lambda x: x["ticker"],
        "culture_analysis": lambda x: x["culture_analysis"]
    }
    | investment_prompt
    | llm
    | StrOutputParser()
)

# ============================================================================
# CHAIN 3: Structured Output - Create Improvement Plan
# ============================================================================
improvement_system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an organizational transformation consultant specializing in 
AI-driven culture change for Private Equity portfolio companies."""
)

improvement_user_prompt = HumanMessagePromptTemplate.from_template(
    """Based on this company analysis:

Company: {company}
Ticker: {ticker}

Culture Analysis:
{culture_analysis}

Investment Recommendation:
{investment_recommendation}

=== YOUR TASK ===
Identify the TOP 3 areas for immediate culture improvement and provide 
specific, actionable initiatives.""",
    input_variables=["company", "ticker", "culture_analysis", "investment_recommendation"]
)

improvement_prompt = ChatPromptTemplate.from_messages(
    [improvement_system_prompt, improvement_user_prompt]
)

# Define structured output format
class ImprovementInitiative(BaseModel):
    priority_area: str = Field(description="The culture dimension to improve")
    current_gap: str = Field(description="What's currently lacking")
    proposed_initiative: str = Field(description="Specific action to take")
    expected_impact: str = Field(description="How this improves AI readiness")

class ImprovementPlan(BaseModel):
    initiatives: list[ImprovementInitiative] = Field(
        description="Top 3 improvement initiatives"
    )
    implementation_timeline: str = Field(
        description="Suggested 90-180 day timeline"
    )

structured_llm = llm.with_structured_output(ImprovementPlan)

chain_three = (
    {
        "company": lambda x: x["company"],
        "ticker": lambda x: x["ticker"],
        "culture_analysis": lambda x: x["culture_analysis"],
        "investment_recommendation": lambda x: x["investment_recommendation"]
    }
    | improvement_prompt
    | structured_llm
)

# ============================================================================
# EXECUTE FULL CHAIN SEQUENCE
# ============================================================================
walmart_reviews = """
Review 1 [4★, Current Employee, Senior Data Analyst, 2024]:
  Pros: "Great push toward data-driven decisions. New analytics dashboards 
         rolled out company-wide. Leadership genuinely embraces AI for 
         supply chain optimization."
  Cons: "Middle management still relies on gut feeling. Change is slow 
         at the store level."

Review 2 [3★, Former Employee, Software Engineer, 2024]:
  Pros: "Invested heavily in ML for demand forecasting. Teams encouraged 
         to experiment with new tools."
  Cons: "Bureaucratic processes slow down innovation. Red tape everywhere 
         for new project approvals. Legacy systems are a constant battle."

Review 3 [5★, Current Employee, Data Scientist, 2025]:
  Pros: "Walmart is not the same company it was 5 years ago. We have a 
         dedicated AI/ML platform team, regular hackathons, and leadership 
         actively promotes data literacy training."
  Cons: "Scale means things move slower than a startup. But direction is right."

Review 4 [2★, Former Employee, Store Manager, 2023]:
  Pros: "Stable company with good benefits."
  Cons: "No idea what AI is at the store level. Corporate talks about 
         digital transformation but it hasn't reached us. Very traditional 
         and hierarchical culture in operations."
"""

# CHAIN 1: Get culture analysis
print("\n[CHAIN 1] Analyzing Culture from Reviews...")
culture_analysis = chain_one.invoke({
    "company": "Walmart",
    "ticker": "WMT",
    "review_count": "4",
    "reviews": walmart_reviews,
})
print(culture_analysis)

# CHAIN 2: Get investment recommendation using culture analysis
print("\n[CHAIN 2] Generating Investment Recommendation...")
investment_rec = chain_two.invoke({
    "company": "Walmart",
    "ticker": "WMT",
    "culture_analysis": culture_analysis
})
print(investment_rec)

# CHAIN 3: Create improvement plan using both previous outputs
print("\n[CHAIN 3] Creating Improvement Plan...")
improvement_plan = chain_three.invoke({
    "company": "Walmart",
    "ticker": "WMT",
    "culture_analysis": culture_analysis,
    "investment_recommendation": investment_rec
})
print("\nIMPROVEMENT INITIATIVES:")
for i, initiative in enumerate(improvement_plan.initiatives, 1):
    print(f"\n{i}. {initiative.priority_area}")
    print(f"   Current Gap: {initiative.current_gap}")
    print(f"   Initiative: {initiative.proposed_initiative}")
    print(f"   Expected Impact: {initiative.expected_impact}")

print(f"\nIMPLEMENTATION TIMELINE:\n{improvement_plan.implementation_timeline}")