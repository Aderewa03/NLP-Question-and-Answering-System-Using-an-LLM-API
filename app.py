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
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  THEME  —  clean blue / futuristic. Pure CSS over Streamlit.
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --glass: rgba(255,255,255,0.035);
  --glass-border: rgba(96,165,250,0.18);
  --accent: #3B82F6;
  --accent-2: #22D3EE;
  --accent-glow: rgba(59,130,246,0.35);
  --text: #E7EEFB;
  --muted: #8A97B8;
}

/* ---------- App background ---------- */
.stApp {
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(37,99,235,0.20), transparent 60%),
    radial-gradient(1000px 700px at 100% 0%, rgba(34,211,238,0.10), transparent 55%),
    linear-gradient(180deg, #070C1C 0%, #05080F 100%);
  background-attachment: fixed;
  color: var(--text);
  font-family: 'Inter', sans-serif;
}

/* ---------- Center + constrain content ---------- */
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 820px;
  padding-top: 3.5rem;
  padding-bottom: 4rem;
}

/* ---------- Hide Streamlit chrome ---------- */
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stStatusWidget"] { display: none; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Typography ---------- */
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--text); letter-spacing: -0.02em; }
[data-testid="stMarkdownContainer"], .stMarkdown, p, li, label { color: var(--text); }

/* ---------- Hero ---------- */
.hero { text-align: center; margin: 0 0 2.4rem 0; }
.hero-title {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.7rem;
  line-height: 1.1; margin: .55rem 0 .5rem 0;
  background: linear-gradient(90deg, #8FBBFF 0%, #22D3EE 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: var(--muted); font-size: 1.03rem; margin: 0; }
.status {
  display: inline-flex; align-items: center; gap: .5rem;
  font-size: .72rem; letter-spacing: .16em; text-transform: uppercase;
  color: #7FE7C4; border: 1px solid rgba(127,231,196,.28);
  padding: .32rem .8rem; border-radius: 999px; background: rgba(127,231,196,.06);
}
.status .dot {
  width: 7px; height: 7px; border-radius: 50%; background: #34D399;
  box-shadow: 0 0 10px #34D399; animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

/* ---------- Field label ---------- */
.field-label {
  font-family: 'JetBrains Mono', monospace; font-size: .72rem;
  letter-spacing: .18em; text-transform: uppercase; color: var(--accent-2);
  margin-bottom: .55rem;
}

/* ---------- Text area ---------- */
.stTextArea textarea {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif; font-size: 1rem;
  padding: 1rem; transition: all .2s ease;
}
.stTextArea textarea:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow), 0 0 26px rgba(59,130,246,.18);
}
.stTextArea textarea::placeholder { color: #5C688A; }

/* ---------- Button ---------- */
.stButton > button {
  background: linear-gradient(135deg, #2E7BFF 0%, #22D3EE 100%);
  color: #04101F !important;
  font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1rem;
  border: none; border-radius: 12px; padding: .7rem 1rem; width: 100%;
  transition: all .2s ease; box-shadow: 0 6px 20px rgba(46,123,255,.32);
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 30px rgba(46,123,255,.5), 0 0 30px rgba(34,211,238,.28);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:focus:not(:active) { color: #04101F !important; }

/* ---------- Glass cards (bordered containers) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--glass);
  border: 1px solid var(--glass-border) !important;
  border-radius: 16px;
  backdrop-filter: blur(8px);
  box-shadow: 0 8px 30px rgba(0,0,0,.25);
}

/* ---------- Result pieces ---------- */
.result-label {
  font-family: 'JetBrains Mono', monospace; font-size: .72rem;
  letter-spacing: .18em; text-transform: uppercase; color: var(--accent-2);
  margin-bottom: .5rem;
}
.processed-chip {
  font-family: 'JetBrains Mono', monospace; color: #A9BCE6; font-size: .92rem;
  background: rgba(59,130,246,.08); border: 1px solid rgba(59,130,246,.18);
  padding: .55rem .85rem; border-radius: 10px; display: inline-block;
}
.model-tag {
  font-family: 'JetBrains Mono', monospace; font-size: .72rem;
  color: var(--muted); margin-top: 1rem; letter-spacing: .04em;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(10,18,41,.92), rgba(6,10,25,.92));
  border-right: 1px solid rgba(96,165,250,.12);
}
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stExpander"] details {
  background: rgba(255,255,255,.03);
  border: 1px solid rgba(96,165,250,.14) !important;
  border-radius: 12px;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
  background: rgba(59,130,246,.08);
  border: 1px solid rgba(59,130,246,.22);
  border-radius: 12px; color: var(--text);
}
[data-testid="stAlert"] * { color: var(--text); }

/* ---------- Dividers ---------- */
hr { border-color: rgba(96,165,250,.12) !important; }

/* ---------- Footer note ---------- */
.footer-note {
  text-align: center; color: var(--muted); font-size: .82rem;
  font-family: 'JetBrains Mono', monospace; letter-spacing: .04em; margin-top: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
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
            max_output_tokens=1024,
        ),
    )
    return response.text or "(No answer returned.)"


# ===== Gemini API call (new SDK, with model fallback) =====
def call_llm_api(user_prompt: str, use_mock: bool = False) -> str:
    if use_mock:
        return mock_llm_response(user_prompt)

    if not genai:
        return "Error: google-genai package not installed. Run: pip install google-genai"
    if not GEMINI_KEY:
        return "Error: GEMINI_API_KEY not found. Set it in Streamlit secrets or environment variables."
    if not client:
        return "Error: Could not initialise the Gemini client. Check your GEMINI_API_KEY."

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
            if "api key" in low or "unauth" in low or "permission" in low:
                return "Error: Authentication failed. Check your GEMINI_API_KEY."
            if "quota" in low or "429" in low or "resource_exhausted" in low:
                return "Error: Rate limit / quota reached. Wait a moment and try again."
            last_err = err
            continue

    return (
        "Error: none of the candidate models were available on your key. "
        "Open Google AI Studio, check which models you have access to, and add "
        f"one to CANDIDATE_MODELS at the top of app.py. Last issue: {last_err}"
    )


# Initialize session state (non-widget values only)
if "history" not in st.session_state:
    st.session_state.history = []

MOCK_DEFAULT = os.getenv("USE_MOCK", "false").lower() == "true"

# ================= UI =================
st.markdown(
    """
<div class="hero">
  <div class="status"><span class="dot"></span> System online</div>
  <h1 class="hero-title">LLM Q&amp;A System</h1>
  <p class="hero-sub">Ask anything and get clear, AI-powered answers in seconds.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---- Input card ----
with st.container(border=True):
    st.markdown('<div class="field-label">Your question</div>', unsafe_allow_html=True)
    user_question = st.text_area(
        "Your question",
        height=120,
        placeholder="e.g., What is machine learning?",
        label_visibility="collapsed",
        key="user_question_input",
    )
    ask = st.button("Get Answer")

# ---- Handle a request ----
if ask:
    if user_question.strip():
        with st.spinner("Thinking..."):
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

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="result-label">Processed question</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="processed-chip">{processed}</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="result-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(answer)
            model_used = st.session_state.get("working_model")
            if model_used and not st.session_state.get("use_mock", MOCK_DEFAULT):
                st.markdown(
                    f'<div class="model-tag">answered by {model_used}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("Please enter a question.")

# ---- Sidebar ----
with st.sidebar:
    st.header("Recent questions")
    use_mock = st.checkbox("Use mock mode (no API key)", value=MOCK_DEFAULT, key="use_mock")
    if use_mock:
        st.info("Using simulated responses.")

    st.markdown("---")
    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"Q{idx + 1}: {item['question'][:38]}...  ({item['timestamp']})"):
                st.markdown(f"**Processed:** {item['processed']}")
                st.markdown(f"**Answer:** {item['answer']}")
    else:
        st.info("No questions yet. Ask something to get started.")

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ---- Footer ----
st.markdown(
    '<div class="footer-note">Built by Aderewa Adesola &nbsp;·&nbsp; Powered by Gemini</div>',
    unsafe_allow_html=True,
)