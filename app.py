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

# Model name. Model IDs change over time — if this ever stops working,
# open Google AI Studio (aistudio.google.com) and copy the current free
# "Flash" model ID here. It's a one-line swap.
MODEL_NAME = "gemini-2.5-flash"

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


# ===== Fixed Gemini API call (new SDK) =====
def call_llm_api(user_prompt: str, use_mock: bool = False) -> str:
    if use_mock:
        return mock_llm_response(user_prompt)

    if not genai:
        return "⚠️ Error: google-genai package not installed. Run: pip install google-genai"
    if not GEMINI_KEY:
        return "⚠️ Error: GEMINI_API_KEY not found. Set it in Streamlit secrets or environment variables."
    if not client:
        return "⚠️ Error: Could not initialise the Gemini client. Check your GEMINI_API_KEY."

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=256,
                temperature=0.7,
            ),
        )
        return response.text or "(No answer returned.)"

    except Exception as e:
        err = str(e)
        low = err.lower()
        if "api key" in low or "unauth" in low or "permission" in low:
            return "⚠️ Error: Authentication failed. Check your GEMINI_API_KEY."
        if "quota" in low or "429" in low or "resource_exhausted" in low:
            return "⚠️ Error: Rate limit / quota reached. Wait a moment and try again."
        if "not found" in low or "404" in low:
            return f"⚠️ Error: Model '{MODEL_NAME}' not found. Update MODEL_NAME to a current one from AI Studio."
        return f"⚠️ Error calling Gemini API: {err}"


# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "use_mock" not in st.session_state:
    st.session_state.use_mock = os.getenv("USE_MOCK", "false").lower() == "true"

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
    )

    if st.button("🚀 Get Answer", use_container_width=True):
        if user_question.strip():
            with st.spinner("Processing your question..."):
                processed = preprocess_question(user_question)
                prompt = build_prompt(processed)
                answer = call_llm_api(prompt, use_mock=st.session_state.use_mock)
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
    st.session_state.use_mock = st.checkbox(
        "Use Mock Mode (No API Key)", value=st.session_state.use_mock
    )
    if st.session_state.use_mock:
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