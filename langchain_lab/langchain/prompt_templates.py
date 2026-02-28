# Prompts for chat agents are at a minimum broken up into three components, those are:
# - System prompt: this provides the instructions to our LLM on how it must behave, what it's objective is, etc.
# - User prompt: this is a user written input.

# - AI prompt: this is the AI generated output. When representing a conversation, previous generations will be inserted back into the next prompt and become part of the broader chat history.

# You are a helpful AI assistant, you will do XYZ.    | SYSTEM PROMPT
# User: Hi, what is the capital of Australia?         | USER PROMPT
# AI: It is Canberra                                  | AI PROMPT
# User: When is the best time to visit?               | USER PROMPT
# ##/

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

llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
print("EXAMPLE : Glassdoor Culture Analyzer")
print("=" * 70)

culture_system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an organizational culture analyst for a Private Equity firm.
You analyze employee reviews to assess a company's culture."""
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
CONFIDENCE: [low/medium/high] — based on review count and recency""",
    input_variables=["company", "ticker", "review_count", "reviews"]
)

culture_prompt = ChatPromptTemplate.from_messages(
    [culture_system_prompt, culture_user_prompt]
)

culture_chain = culture_prompt | llm | StrOutputParser()

# --- Run Example: Analyze Walmart's Culture Reviews ---
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

result = culture_chain.invoke({
    "company": "Walmart",
    "ticker": "WMT",
    "review_count": "4",
    "reviews": walmart_reviews,
})
print(result)
print()