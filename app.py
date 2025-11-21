import streamlit as st
import os
import string
import time

# Try to import Gemini client
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ============= PROMPT TEMPLATES =============
SYSTEM_PROMPT = """You are a helpful assistant. Provide concise answers and cite sources when possible."""
USER_PROMPT_BASIC = "{processed_question}"
USER_PROMPT_ENHANCED = """Please answer the following question concisely:

Question: {processed_question}

Provide a clear, factual answer. If applicable, cite reliable sources."""
# ============================================

# Page configuration
st.set_page_config(
    page_title="LLM Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (kept as original)
st.markdown("""<your original CSS here>""", unsafe_allow_html=True)

# Load Gemini API key from Streamlit secrets or env
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if genai and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception:
        pass  # ignore library version differences

# ================= Helper functions =================
def preprocess_question(question: str) -> str:
    question_lower = question.lower()
    translator = str.maketrans('', '', string.punctuation)
    question_no_punct = question_lower.translate(translator)
    return ' '.join(question_no_punct.split())

def build_prompt(processed_question: str, use_enhanced: bool = False) -> list:
    user_content = USER_PROMPT_ENHANCED.format(processed_question=processed_question) if use_enhanced else USER_PROMPT_BASIC.format(processed_question=processed_question)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}]

def call_llm_api(messages: list, use_mock: bool = False) -> str:
    if use_mock:
        return mock_llm_response(messages[-1]["content"])

    if not GEMINI_KEY:
        return "⚠️ Error: GEMINI_API_KEY not found. Set it in Streamlit secrets or environment variables."
    if not genai:
        return "⚠️ Error: google-generativeai package not installed. Run: pip install google-generativeai"

    user_prompt = messages[-1]["content"]

    try:
        response = genai.generate(model="gemini-pro", prompt=user_prompt, max_output_tokens=256, temperature=0.7)
        # extract text from response safely
        if hasattr(response, "text"):
            return response.text
        if isinstance(response, dict):
            if "candidates" in response and len(response["candidates"]) > 0:
                cand = response["candidates"][0]
                if isinstance(cand, dict) and "content" in cand:
                    return cand["content"]
            if "output" in response:
                out = response["output"]
                if isinstance(out, str):
                    return out
                if isinstance(out, list):
                    return " ".join([item["content"] if isinstance(item, dict) and "content" in item else str(item) for item in out])
        return str(response)
    except Exception as e:
        err = str(e)
        if "authentication" in err.lower() or "unauth" in err.lower():
            return "⚠️ Error: Authentication failed. Check your GEMINI_API_KEY."
        if "quota" in err.lower() or "billing" in err.lower():
            return "⚠️ Error: Quota/billing issue. Check your Google Cloud / AI Studio usage."
        return f"⚠️ Error calling Gemini API: {err}"

def mock_llm_response(question: str) -> str:
    q = question.lower()
    if "capital" in q or "city" in q:
        return "Mock Response: For example, the capital of France is Paris."
    if "python" in q or "code" in q:
        return "Mock Response: Python is a high-level programming language used for many tasks."
    return "Mock Response: This is a simulated answer."

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'use_mock' not in st.session_state:
    st.session_state.use_mock = os.getenv("USE_MOCK", "false").lower() == "true"

# ========== UI (kept as your original) ==========
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🤖 LLM Q&A System</h1>
    <p class="header-subtitle">Ask any question and get AI-powered answers instantly</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Ask Your Question")
    user_question = st.text_area("Type your question here:", height=100, placeholder="e.g., What is machine learning?", label_visibility="collapsed")
    if st.button("🚀 Get Answer", use_container_width=True):
        if user_question.strip():
            with st.spinner("Processing your question..."):
                processed = preprocess_question(user_question)
                messages = build_prompt(processed)
                answer = call_llm_api(messages, use_mock=st.session_state.use_mock)
                st.session_state.history.insert(0, {"question": user_question,"processed": processed,"answer": answer,"timestamp": time.strftime("%H:%M:%S")})
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop()
            st.markdown("---")
            st.markdown(f'<div class="result-card"><div class="result-label">Processed Question</div><div class="result-text">{processed}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-card"><div class="result-label">Answer</div><div class="result-text">{answer}</div></div>', unsafe_allow_html=True)
            st.button("📋 Copy Answer", on_click=lambda: st.write("Answer copied!"))
        else:
            st.warning("⚠️ Please enter a question.")

with st.sidebar:
    st.header("📜 Recent Questions")
    if st.checkbox("Use Mock Mode (No API Key)", value=st.session_state.use_mock):
        st.session_state.use_mock = True
        st.info("🔄 Using simulated responses")
    else:
        st.session_state.use_mock = False
    st.markdown("---")
    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"Q{idx+1}: {item['question'][:40]}... ({item['timestamp']})"):
                st.markdown(f"**Processed:** {item['processed']}")
                st.markdown(f"**Answer:** {item['answer']}")
    else:
        st.info("No questions asked yet. Start by asking something!")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
    CSC415/CSC331 AI Project 2 | Built with Streamlit & Gemini
</div>
""", unsafe_allow_html=True)
