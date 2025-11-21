import streamlit as st
import os
import string
from openai import OpenAI
import time

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

# Custom CSS for styling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Main container */
    .main {
        background-color: #F9FAFB;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #4F46E5 0%, #8B5CF6 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #E0E7FF;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Result card styling */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border-left: 4px solid #4F46E5;
    }
    
    .result-label {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #4F46E5;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .result-text {
        color: #1F2937;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #8B5CF6 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* History item */
    .history-item {
        background: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        border-left: 3px solid #8B5CF6;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_question(question: str) -> str:
    """
    Preprocess the input question:
    - Convert to lowercase
    - Tokenize by whitespace
    - Remove punctuation
    """
    question_lower = question.lower()
    translator = str.maketrans('', '', string.punctuation)
    question_no_punct = question_lower.translate(translator)
    tokens = question_no_punct.split()
    processed = ' '.join(tokens)
    return processed

def build_prompt(processed_question: str, use_enhanced: bool = False) -> list:
    """
    Build the prompt messages for the LLM API using templates.
    """
    if use_enhanced:
        user_content = USER_PROMPT_ENHANCED.format(processed_question=processed_question)
    else:
        user_content = USER_PROMPT_BASIC.format(processed_question=processed_question)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    return messages

def call_llm_api(messages: list, use_mock: bool = False) -> str:
    """
    Send prompt to OpenAI API and return response.
    """
    if use_mock:
        return mock_llm_response(messages[-1]["content"])
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ Error: OPENAI_API_KEY not found. Please set it in Streamlit secrets or environment variables."
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            timeout=30.0
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        if "timeout" in str(e).lower():
            return "⚠️ Error: Request timed out. Please try again."
        elif "rate_limit" in str(e).lower():
            return "⚠️ Error: Rate limit exceeded. Please wait and try again."
        elif "authentication" in str(e).lower():
            return "⚠️ Error: Invalid API key."
        else:
            return f"⚠️ Error: {str(e)}"

def mock_llm_response(question: str) -> str:
    """
    Fallback mock function for offline testing.
    """
    question_lower = question.lower()
    
    if "capital" in question_lower or "city" in question_lower:
        return "Mock Response: The capital varies by country. For example, the capital of France is Paris."
    elif "python" in question_lower or "code" in question_lower:
        return "Mock Response: Python is a high-level programming language known for readability and versatility."
    elif "weather" in question_lower or "temperature" in question_lower:
        return "Mock Response: I cannot access real-time weather data. Please check a weather service."
    else:
        return f"Mock Response: This is a simulated answer to your question."

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

if 'use_mock' not in st.session_state:
    st.session_state.use_mock = os.getenv("USE_MOCK", "false").lower() == "true"

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🤖 LLM Q&A System</h1>
    <p class="header-subtitle">Ask any question and get AI-powered answers instantly</p>
</div>
""", unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask Your Question")
    
    # Question input
    user_question = st.text_area(
        "Type your question here:",
        height=100,
        placeholder="e.g., What is machine learning?",
        label_visibility="collapsed"
    )
    
    # Send button
    if st.button("🚀 Get Answer", use_container_width=True):
        if user_question.strip():
            with st.spinner("Processing your question..."):
                # Preprocess
                processed = preprocess_question(user_question)
                
                # Build prompt
                messages = build_prompt(processed)
                
                # Get answer
                answer = call_llm_api(messages, use_mock=st.session_state.use_mock)
                
                # Add to history (keep last 5)
                st.session_state.history.insert(0, {
                    "question": user_question,
                    "processed": processed,
                    "answer": answer,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop()
            
            # Display results
            st.markdown("---")
            st.markdown(f'<div class="result-card"><div class="result-label">Processed Question</div><div class="result-text">{processed}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-card"><div class="result-label">Answer</div><div class="result-text">{answer}</div></div>', unsafe_allow_html=True)
            
            # Copy button
            st.button("📋 Copy Answer", on_click=lambda: st.write("Answer copied!"))
        else:
            st.warning("⚠️ Please enter a question.")

# Sidebar - History
with st.sidebar:
    st.header("📜 Recent Questions")
    
    # Mock mode toggle
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
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
    CSC415/CSC331 AI Project 2 | Built with Streamlit & OpenAI
</div>
""", unsafe_allow_html=True)