import os
import litellm
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

# ✅ Explicitly set keys in os.environ so LiteLLM can find them
os.environ["OPENAI_API_KEY"]    = os.getenv("OPENAI_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")

def track_cost_callback(
    kwargs,
    completion_response,
    start_time, end_time
):
    try:
        # ✅ Skip mid-stream chunks — only log when cost is available
        response_cost = kwargs.get("response_cost")
        if response_cost is None:
            return

        model    = kwargs.get("model", "unknown")
        duration = (end_time - start_time).total_seconds()
        tokens   = getattr(getattr(completion_response, "usage", None), "total_tokens", "N/A")

        print(f"\n📊 Cost Report")
        print(f"  Model    : {model}")
        print(f"  Cost     : ${response_cost:.6f}")
        print(f"  Duration : {duration:.2f}s")
        print(f"  Tokens   : {tokens}")
    except Exception as e:
        print(f"Callback error: {e}")

litellm.success_callback = [track_cost_callback]

# ── OpenAI ─────────────────────────────────────────────────────────
print("🔵 OpenAI - GPT-4o Mini")

openai_response = completion(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a PE analyst."},
        {"role": "user",   "content": "What signals indicate strong AI readiness?"}
    ],
    stream=True
)

for chunk in openai_response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# ── Anthropic ──────────────────────────────────────────────────────
print("\n" + "=" * 40)
print("🟠 Anthropic - Claude Haiku")

claude_response = completion(
    model="claude-haiku-4-5-20251001",
    messages=[
        {"role": "system", "content": "You are a PE analyst."},
        {"role": "user",   "content": "What signals indicate strong AI readiness?"}
    ],
    stream=True
)

for chunk in claude_response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)