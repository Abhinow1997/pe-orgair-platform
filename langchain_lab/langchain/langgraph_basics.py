# ============================================================================
# LangGraph Fundamentals — Basic Concepts
# ============================================================================
# 
# LangGraph = LangChain + State + Decisions + Loops
#
# Think of it this way:
#   LangChain  = Assembly line (A → B → C → done)
#   LangGraph  = Flowchart    (A → B → if X go back to A, else go to C)
#
# Prerequisites:
#   pip install langgraph langchain langchain-anthropic langchain-core python-dotenv
# ============================================================================

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================================
# CONCEPT 1: STATE — The Shared Memory
# ============================================================================
# 
# In LangChain, data flows through pipes and is GONE after each step.
# In LangGraph, there's a STATE OBJECT that every step can read and write to.
# 
# Think of it like a shared whiteboard in a meeting room.
# Every person (node) can walk up, read what's there, and add their notes.
# The whiteboard persists throughout the entire meeting.
#
# In Python terms, state is just a TypedDict — a dictionary with defined keys.
# ============================================================================

print("=" * 70)
print("CONCEPT 1: STATE — The Shared Whiteboard")
print("=" * 70)

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# This is our state — the whiteboard everyone shares
class SimpleState(TypedDict):
    name: str                  # A regular field — gets REPLACED on update
    age: int                   # Same — replaced on update
    messages: Annotated[       # Special! Uses a REDUCER (add_messages)
        list,                  # This means new messages get APPENDED, not replaced
        add_messages
    ]

# How state updates work:
# 
#   Regular field:
#     State has: {"name": "Alice"}
#     Node returns: {"name": "Bob"}
#     Result: {"name": "Bob"}          ← REPLACED
#
#   Field with add_messages reducer:
#     State has: {"messages": [msg1]}
#     Node returns: {"messages": [msg2]}
#     Result: {"messages": [msg1, msg2]}  ← APPENDED
#
# Why does this matter?
# Chat history should GROW (append), not get replaced.
# But a score should be REPLACED with the latest calculation.

print("State is a TypedDict — a dictionary with typed keys.")
print("Regular fields get replaced. Fields with reducers get appended.")
print()


# ============================================================================
# CONCEPT 2: NODES — The Workers
# ============================================================================
#
# A node is just a Python function that:
#   1. Receives the full state
#   2. Does some work
#   3. Returns a dict with the fields it wants to UPDATE
#
# The node does NOT need to return the entire state.
# If it only changes "score", it returns {"score": 85}.
# All other fields stay as they are.
#
# Think of each node as a person at the whiteboard:
#   - They read what they need
#   - They write their contribution
#   - They sit back down
# ============================================================================

print("=" * 70)
print("CONCEPT 2: NODES — The Workers")
print("=" * 70)

# Let's define a state for a simple greeting pipeline
class GreetingState(TypedDict):
    name: str
    greeting: str
    farewell: str

# Node 1: Creates a greeting
def create_greeting(state: GreetingState) -> dict:
    """This node reads 'name' and writes 'greeting'."""
    name = state["name"]
    return {"greeting": f"Hello, {name}! Welcome to LangGraph."}

# Node 2: Creates a farewell
def create_farewell(state: GreetingState) -> dict:
    """This node reads 'name' and writes 'farewell'."""
    name = state["name"]
    return {"farewell": f"Goodbye, {name}! See you next time."}

# Notice:
# - create_greeting only returns {"greeting": ...} — it doesn't touch "farewell"
# - create_farewell only returns {"farewell": ...} — it doesn't touch "greeting"
# - Each node does ONE job and updates only what it needs to

print("Nodes are just functions: receive state → do work → return updates.")
print()


# ============================================================================
# CONCEPT 3: EDGES — The Connections
# ============================================================================
#
# Edges tell LangGraph: "After this node finishes, go to that node."
#
# Two types:
#   1. Normal edge:      A → B (always)
#   2. Conditional edge:  A → B or C (depends on state)
#
# This is where LangGraph becomes powerful.
# LangChain can only do: A → B → C → done
# LangGraph can do:       A → B → (if low confidence, go back to A)
# ============================================================================

print("=" * 70)
print("CONCEPT 3: EDGES — Connecting the Nodes")
print("=" * 70)

from langgraph.graph import StateGraph, START, END

# BUILD THE GRAPH

# Step 1: Create a graph builder with our state type
builder = StateGraph(GreetingState)

# Step 2: Add nodes (register the functions)
builder.add_node("greeter", create_greeting)
builder.add_node("fareweller", create_farewell)

# Step 3: Add edges (define the flow)
builder.add_edge(START, "greeter")           # Start → greeter
builder.add_edge("greeter", "fareweller")    # greeter → fareweller
builder.add_edge("fareweller", END)          # fareweller → End

# Step 4: Compile the graph (makes it executable)
graph = builder.compile()

# Step 5: Run it!
result = graph.invoke({"name": "Abhinav"})
print(f"Input:    name = 'Abhinav'")
print(f"Greeting: {result['greeting']}")
print(f"Farewell: {result['farewell']}")
print()

# What just happened:
#
#   START → greeter → fareweller → END
#
#   1. State starts as: {"name": "Abhinav", "greeting": "", "farewell": ""}
#   2. greeter runs:    {"name": "Abhinav", "greeting": "Hello, Abhinav!...", "farewell": ""}
#   3. fareweller runs: {"name": "Abhinav", "greeting": "Hello, Abhinav!...", "farewell": "Goodbye, Abhinav!..."}
#   4. Done!


# ============================================================================
# CONCEPT 4: CONDITIONAL EDGES — The Decision Maker
# ============================================================================
#
# This is THE reason LangGraph exists.
#
# A conditional edge is a fork in the road:
#   "Look at the current state. Based on what you see, pick the next node."
#
# Real-world example from CS3:
#   After scoring a company, check the confidence.
#   If confidence < 0.7 → go back and gather more evidence
#   If confidence >= 0.7 → proceed to final calculation
#
# LangChain CANNOT do this. It always follows the same path.
# ============================================================================

print("=" * 70)
print("CONCEPT 4: CONDITIONAL EDGES — Making Decisions")
print("=" * 70)

class ScoreState(TypedDict):
    number: int
    category: str
    explanation: str

# Node 1: Categorize the number
def categorize(state: ScoreState) -> dict:
    n = state["number"]
    if n >= 75:
        return {"category": "high"}
    elif n >= 50:
        return {"category": "medium"}
    else:
        return {"category": "low"}

# Node 2a: Handle high scores
def handle_high(state: ScoreState) -> dict:
    return {"explanation": f"{state['number']} is HIGH — industry leader!"}

# Node 2b: Handle medium scores
def handle_medium(state: ScoreState) -> dict:
    return {"explanation": f"{state['number']} is MEDIUM — competitive but room to grow."}

# Node 2c: Handle low scores
def handle_low(state: ScoreState) -> dict:
    return {"explanation": f"{state['number']} is LOW — needs significant investment."}

# The ROUTING FUNCTION — looks at state, returns the name of the next node
def route_by_category(state: ScoreState) -> str:
    """This function decides which node to go to next."""
    if state["category"] == "high":
        return "handle_high"
    elif state["category"] == "medium":
        return "handle_medium"
    else:
        return "handle_low"

# Build the graph
builder2 = StateGraph(ScoreState)

builder2.add_node("categorize", categorize)
builder2.add_node("handle_high", handle_high)
builder2.add_node("handle_medium", handle_medium)
builder2.add_node("handle_low", handle_low)

builder2.add_edge(START, "categorize")

# THE CONDITIONAL EDGE — this is the magic
builder2.add_conditional_edges(
    "categorize",              # After this node finishes...
    route_by_category,         # ...call this function to decide where to go
    {                          # ...these are the possible destinations
        "handle_high": "handle_high",
        "handle_medium": "handle_medium",
        "handle_low": "handle_low",
    }
)

builder2.add_edge("handle_high", END)
builder2.add_edge("handle_medium", END)
builder2.add_edge("handle_low", END)

graph2 = builder2.compile()

# Test with different scores
for score in [85, 60, 30]:
    result = graph2.invoke({"number": score})
    print(f"  Score {score}: {result['explanation']}")

print()
# Visual:
#
#   START → categorize
#                │
#         ┌──────┼──────┐
#         ▼      ▼      ▼
#       high   medium   low
#         │      │      │
#         └──────┼──────┘
#                ▼
#               END


# ============================================================================
# CONCEPT 5: LOOPS — Going Back for More
# ============================================================================
#
# This is what makes LangGraph truly different from LangChain.
# A node can LOOP BACK to a previous node based on a condition.
#
# Real-world scenario:
#   You're scoring a company but confidence is low (not enough evidence).
#   Instead of returning a bad score, the system loops back to gather
#   more evidence, then re-scores. It keeps doing this until confidence
#   is high enough OR it hits a maximum number of iterations.
#
# LangChain literally cannot do this. Data only flows forward.
# ============================================================================

print("=" * 70)
print("CONCEPT 5: LOOPS — The Confidence Retry Pattern")
print("=" * 70)

import random

class RetryState(TypedDict):
    target: int            # The number we're trying to guess close to
    current_guess: int     # Our latest guess
    attempts: int          # How many tries so far
    max_attempts: int      # Safety limit
    done: bool             # Are we close enough?

def make_guess(state: RetryState) -> dict:
    """Generate a random guess between 1 and 100."""
    guess = random.randint(1, 100)
    attempts = state.get("attempts", 0) + 1
    print(f"    Attempt {attempts}: Guessed {guess} (target: {state['target']})")
    return {"current_guess": guess, "attempts": attempts}

def evaluate_guess(state: RetryState) -> dict:
    """Check if the guess is close enough (within 15)."""
    diff = abs(state["current_guess"] - state["target"])
    close_enough = diff <= 15
    return {"done": close_enough}

def format_result(state: RetryState) -> dict:
    """Format the final answer."""
    diff = abs(state["current_guess"] - state["target"])
    return {
        "current_guess": state["current_guess"],  # Keep the guess
    }

# THE ROUTING FUNCTION — decides: try again or stop?
def should_retry(state: RetryState) -> str:
    if state["done"]:
        return "format_result"                   # Close enough! Move on.
    elif state["attempts"] >= state["max_attempts"]:
        return "format_result"                   # Hit max tries. Stop.
    else:
        return "make_guess"                      # Not close enough. TRY AGAIN.

# Build the graph WITH A LOOP
builder3 = StateGraph(RetryState)

builder3.add_node("make_guess", make_guess)
builder3.add_node("evaluate_guess", evaluate_guess)
builder3.add_node("format_result", format_result)

builder3.add_edge(START, "make_guess")
builder3.add_edge("make_guess", "evaluate_guess")

# CONDITIONAL EDGE WITH LOOP-BACK
builder3.add_conditional_edges(
    "evaluate_guess",
    should_retry,
    {
        "make_guess": "make_guess",        # ← LOOP BACK to make_guess!
        "format_result": "format_result",  # ← Move forward
    }
)

builder3.add_edge("format_result", END)

graph3 = builder3.compile()

# Run it — watch it retry until it gets close!
print("  Trying to guess close to 50 (within 15):")
result = graph3.invoke({
    "target": 50,
    "current_guess": 0,
    "attempts": 0,
    "max_attempts": 10,
    "done": False,
})
print(f"  Final guess: {result['current_guess']} after {result['attempts']} attempts")
print(f"  Close enough: {result['done']}")
print()

# Visual:
#
#   START → make_guess → evaluate_guess
#               ▲              │
#               │         ┌────┴────┐
#               │         │         │
#               │       done?     not done
#               │         │     & attempts
#               │         │      < max
#               │         ▼         │
#               │    format_result  │
#               │         │         │
#               │         ▼         │
#               │        END        │
#               └───────────────────┘
#                    LOOP BACK!


# ============================================================================
# CONCEPT 6: USING LLMs INSIDE NODES
# ============================================================================
#
# So far our nodes used plain Python. Now let's put an LLM inside a node.
# This is where LangChain (prompts, models) meets LangGraph (state, decisions).
#
# Pattern: LangChain does the LLM work INSIDE LangGraph nodes.
# ============================================================================

print("=" * 70)
print("CONCEPT 6: LLMs Inside Nodes")
print("=" * 70)

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

class JokeState(TypedDict):
    topic: str
    joke: str
    is_funny: bool
    feedback: str

# Node 1: Generate a joke using the LLM
def generate_joke(state: JokeState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        "Tell me a short, clean joke about {topic}. Just the joke, nothing else."
    )
    chain = prompt | llm | StrOutputParser()
    joke = chain.invoke({"topic": state["topic"]})
    print(f"  Generated joke: {joke[:80]}...")
    return {"joke": joke}

# Node 2: Judge the joke using the LLM
def judge_joke(state: JokeState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        """Rate this joke on funniness. Reply with ONLY "funny" or "not funny".
        
Joke: {joke}"""
    )
    chain = prompt | llm | StrOutputParser()
    verdict = chain.invoke({"joke": state["joke"]})
    is_funny = "funny" in verdict.lower() and "not funny" not in verdict.lower()
    print(f"  Judge says: {verdict.strip()} → is_funny={is_funny}")
    return {"is_funny": is_funny}

# Node 3: Provide feedback
def give_feedback(state: JokeState) -> dict:
    if state["is_funny"]:
        return {"feedback": "Great joke! The audience loved it."}
    else:
        return {"feedback": "Tough crowd. Maybe try a different angle next time."}

# Routing: if not funny AND we haven't retried, try again
attempt_count = {"count": 0}  # Simple counter (in production, put this in state)

def should_retry_joke(state: JokeState) -> str:
    attempt_count["count"] += 1
    if not state["is_funny"] and attempt_count["count"] < 3:
        return "generate_joke"    # Try again!
    return "give_feedback"        # Accept the result

# Build the graph
builder4 = StateGraph(JokeState)

builder4.add_node("generate_joke", generate_joke)
builder4.add_node("judge_joke", judge_joke)
builder4.add_node("give_feedback", give_feedback)

builder4.add_edge(START, "generate_joke")
builder4.add_edge("generate_joke", "judge_joke")
builder4.add_conditional_edges(
    "judge_joke",
    should_retry_joke,
    {"generate_joke": "generate_joke", "give_feedback": "give_feedback"}
)
builder4.add_edge("give_feedback", END)

graph4 = builder4.compile()

result = graph4.invoke({"topic": "data engineering"})
print(f"  Final joke: {result['joke'][:100]}...")
print(f"  Feedback: {result['feedback']}")
print()


# ============================================================================
# CONCEPT 7: CHECKPOINTING — Memory Across Conversations
# ============================================================================
#
# Without checkpointing, every graph.invoke() starts fresh.
# With checkpointing, the graph REMEMBERS previous runs.
#
# This enables:
#   - Multi-turn conversations ("Score NVIDIA" → "Now compare to JPM")
#   - Pause and resume workflows
#   - Debugging (replay from any checkpoint)
#
# You give each conversation a thread_id.
# Same thread_id = same memory.
# Different thread_id = fresh start.
# ============================================================================

print("=" * 70)
print("CONCEPT 7: CHECKPOINTING — Memory Across Runs")
print("=" * 70)

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chat_node(state: ChatState) -> dict:
    """Simple chat node that calls the LLM with full history."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build a simple chat graph
chat_builder = StateGraph(ChatState)
chat_builder.add_node("chat", chat_node)
chat_builder.add_edge(START, "chat")
chat_builder.add_edge("chat", END)

# COMPILE WITH CHECKPOINTER — this enables memory
memory = MemorySaver()
chat_graph = chat_builder.compile(checkpointer=memory)

# Conversation 1 — thread "demo-001"
config = {"configurable": {"thread_id": "demo-001"}}

result1 = chat_graph.invoke(
    {"messages": [HumanMessage(content="My name is Abhinav. Remember that.")]},
    config
)
print(f"  Turn 1: {result1['messages'][-1].content[:100]}...")

# Conversation 2 — SAME thread, so it REMEMBERS
result2 = chat_graph.invoke(
    {"messages": [HumanMessage(content="What is my name?")]},
    config  # Same thread_id!
)
print(f"  Turn 2: {result2['messages'][-1].content[:100]}...")

# Conversation 3 — DIFFERENT thread, fresh start
config_new = {"configurable": {"thread_id": "demo-002"}}
result3 = chat_graph.invoke(
    {"messages": [HumanMessage(content="What is my name?")]},
    config_new  # Different thread!
)
print(f"  Turn 3 (new thread): {result3['messages'][-1].content[:100]}...")
print()

# Key takeaway:
#   thread_id "demo-001" → remembers "Abhinav"
#   thread_id "demo-002" → has no idea who you are


# ============================================================================
# CONCEPT 8: PREBUILT REACT AGENT — The Easy Path
# ============================================================================
#
# If you just want an agent that calls tools and loops automatically,
# LangGraph provides a prebuilt one. No need to build the graph yourself.
#
# create_react_agent = "Give me an LLM and tools, I'll handle the rest"
#
# It automatically:
#   - Sends messages to the LLM
#   - If LLM wants to call a tool → executes it → loops back to LLM
#   - If LLM has no more tool calls → returns the final answer
# ============================================================================

print("=" * 70)
print("CONCEPT 8: PREBUILT REACT AGENT")
print("=" * 70)

from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def get_company_score(ticker: str) -> str:
    """Get the AI readiness score for a company."""
    scores = {
        "NVDA": "NVIDIA: Score 90/100 (Leading)",
        "JPM": "JPMorgan: Score 70/100 (Competitive)",
        "WMT": "Walmart: Score 60/100 (Competitive)",
        "GE": "GE: Score 50/100 (Developing)",
        "DG": "Dollar General: Score 38/100 (Developing)",
    }
    return scores.get(ticker.upper(), f"No data for {ticker}")

# Create the agent — one line!
agent = create_agent(
    model=llm,
    tools=[add_numbers, multiply_numbers, get_company_score],
)

# The agent DECIDES which tools to use
result = agent.invoke({
    "messages": [HumanMessage(content="What's NVIDIA's AI readiness score? Also what's 45 + 37?")]
})

# Print the final response
final_message = result["messages"][-1]
print(f"  Agent response: {final_message.content[:200]}...")
print()

# What happened internally:
#
#   1. LLM sees the question
#   2. LLM decides: "I need get_company_score for NVDA and add_numbers for 45+37"
#   3. LangGraph executes both tools
#   4. Results go back to LLM
#   5. LLM formats the final answer
#   6. No more tool calls → done!


# ============================================================================
# SUMMARY: THE 8 CONCEPTS
# ============================================================================
#
# 1. STATE        — Shared whiteboard (TypedDict) all nodes read/write
# 2. NODES        — Worker functions that update state
# 3. EDGES        — Connections between nodes (A → B)
# 4. CONDITIONAL  — Fork in the road based on state
#    EDGES          ("if score > 75 go to handle_high, else handle_low")
# 5. LOOPS        — Go BACK to a previous node (impossible in LangChain)
#                   ("confidence too low? gather more evidence and re-score")
# 6. LLMs IN      — Put LangChain chains inside LangGraph nodes
#    NODES          (best of both worlds)
# 7. CHECKPOINTS  — Memory across conversations using thread_id
#                   ("What did we discuss about NVIDIA?")
# 8. REACT AGENT  — Prebuilt agent that handles tool-calling loops for you
#                   (one-liner for common use cases)
#
# When to use what:
#   Simple prompt → response:           Just use LangChain
#   Fixed pipeline (A → B → C):         LangChain LCEL
#   Decisions or loops needed:           LangGraph StateGraph
#   Tool-calling agent:                  LangGraph create_react_agent
#   Multi-turn with memory:             LangGraph + Checkpointer
# ============================================================================