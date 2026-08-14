import streamlit as st
import os
import string
import time

# New unified Google GenAI SDK (replaces the deprecated google-generativeai)
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ============= PROMPT TEMPLATES =============
SYSTEM_PROMPT = "You are a helpful assistant. Provide concise answers and cite sources when possible."
USER_PROMPT_BASIC = "{processed_question}"
USER_PROMPT_ENHANCED = """Please answer the following question concisely:

Question: {processed_question}

Provide a clear, factual answer. If applicable, cite reliable sources."""
# ============================================

# Google retires and renames Gemini models every few months. Rather than
# hardcode one (which breaks when it's retired), the app tries these in order
# and uses the first one your API key can access. 'gemini-flash-latest' always
# resolves to a current model, so it's the safety net at the end.
# If you want to force a specific model, put it first in this list.
CANDIDATE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

# Page configuration
st.set_page_config(
    page_title="LLM Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Safely read the API key (works whether or not secrets.toml exists) ----
def get_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


GEMINI_KEY = get_api_key()

# Create the client once, reuse it across requests.
client = None
if genai and GEMINI_KEY:
    try:
        client = genai.Client(api_key=GEMINI_KEY)
    except Exception:
        client = None


# ================= Helper functions =================
def preprocess_question(question: str) -> str:
    question_lower = question.lower()
    translator = str.maketrans("", "", string.punctuation)
    question_no_punct = question_lower.translate(translator)
    return " ".join(question_no_punct.split())


def build_prompt(processed_question: str, use_enhanced: bool = False) -> str:
    if use_enhanced:
        return USER_PROMPT_ENHANCED.format(processed_question=processed_question)
    return USER_PROMPT_BASIC.format(processed_question=processed_question)


def mock_llm_response(question: str) -> str:
    q = question.lower()
    if "capital" in q or "city" in q:
        return "Mock Response: For example, the capital of France is Paris."
    if "python" in q or "code" in q:
        return "Mock Response: Python is a high-level programming language used for many tasks."
    return "Mock Response: This is a simulated answer."


def _generate_once(model_name: str, user_prompt: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=256,
        ),
    )
    return response.text or "(No answer returned.)"


# ===== Gemini API call (new SDK, with model fallback) =====
def call_llm_api(user_prompt: str, use_mock: bool = False) -> str:
    if use_mock:
        return mock_llm_response(user_prompt)

    if not genai:
        return "⚠️ Error: google-genai package not installed. Run: pip install google-genai"
    if not GEMINI_KEY:
        return "⚠️ Error: GEMINI_API_KEY not found. Set it in Streamlit secrets or environment variables."
    if not client:
        return "⚠️ Error: Could not initialise the Gemini client. Check your GEMINI_API_KEY."

    # Try a model that already worked this session first, then the rest.
    models_to_try = list(CANDIDATE_MODELS)
    cached = st.session_state.get("working_model")
    if cached:
        models_to_try = [cached] + [m for m in models_to_try if m != cached]

    last_err = ""
    for model_name in models_to_try:
        try:
            answer = _generate_once(model_name, user_prompt)
            st.session_state["working_model"] = model_name  # remember what worked
            return answer
        except Exception as e:
            err = str(e)
            low = err.lower()
            # Hard stops — no point trying other models for these.
            if "api key" in low or "unauth" in low or "permission" in low:
                return "⚠️ Error: Authentication failed. Check your GEMINI_API_KEY."
            if "quota" in low or "429" in low or "resource_exhausted" in low:
                return "⚠️ Error: Rate limit / quota reached. Wait a moment and try again."
            # Otherwise (model not found, etc.) remember and try the next one.
            last_err = err
            continue

    return (
        "⚠️ Error: none of the candidate models were available on your key. "
        "Open Google AI Studio, check which models you have access to, and add "
        f"one to CANDIDATE_MODELS at the top of app.py. Last issue: {last_err}"
    )


# Initialize session state (non-widget values only)
if "history" not in st.session_state:
    st.session_state.history = []

# Default for the mock toggle, taken from an optional env var.
MOCK_DEFAULT = os.getenv("USE_MOCK", "false").lower() == "true"

# ================= UI =================
st.title("🤖 LLM Q&A System")
st.caption("Ask any question and get AI-powered answers instantly")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Ask Your Question")
    user_question = st.text_area(
        "Type your question here:",
        height=100,
        placeholder="e.g., What is machine learning?",
        label_visibility="collapsed",
        key="user_question_input",
    )

    if st.button("🚀 Get Answer", use_container_width=True):
        if user_question.strip():
            with st.spinner("Processing your question..."):
                processed = preprocess_question(user_question)
                prompt = build_prompt(processed)
                answer = call_llm_api(
                    prompt, use_mock=st.session_state.get("use_mock", MOCK_DEFAULT)
                )
                st.session_state.history.insert(
                    0,
                    {
                        "question": user_question,
                        "processed": processed,
                        "answer": answer,
                        "timestamp": time.strftime("%H:%M:%S"),
                    },
                )
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop()

            st.markdown("---")
            st.markdown("**Processed Question**")
            st.write(processed)
            st.markdown("**Answer**")
            st.write(answer)
        else:
            st.warning("⚠️ Please enter a question.")

with st.sidebar:
    st.header("📜 Recent Questions")
    # The checkbox owns its own state via key="use_mock" — no manual assignment.
    use_mock = st.checkbox(
        "Use Mock Mode (No API Key)", value=MOCK_DEFAULT, key="use_mock"
    )
    if use_mock:
        st.info("🔄 Using simulated responses")

    st.markdown("---")
    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"Q{idx + 1}: {item['question'][:40]}... ({item['timestamp']})"):
                st.markdown(f"**Processed:** {item['processed']}")
                st.markdown(f"**Answer:** {item['answer']}")
    else:
        st.info("No questions asked yet. Start by asking something!")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
st.caption("CSC415/CSC331 AI Project 2 | Built with Streamlit & Gemini")