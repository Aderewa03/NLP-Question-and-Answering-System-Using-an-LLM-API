# LLM Question & Answer System

**Course:** CSC415 - Artificial Intelligence  
**Project:** Project 2 - NLP Question-and-Answering System Using an LLM API  
**Student Name:** Aderewa Adesola
**Matric Number:** 22CG031811
**Date:** 21 November 2025

---

## Project Overview

This project implements a Natural Language Processing (NLP) Question-and-Answering system powered by OpenAI's GPT models. Users can ask questions via a Command Line Interface (CLI) or a modern Web GUI built with Streamlit.

**Features:**
- Text preprocessing (lowercase, tokenization, punctuation removal)
- Integration with OpenAI API (GPT-3.5-turbo)
- Mock mode for testing without API key
- Web interface with question history (last 5 Q&As)
- Responsive and modern UI design
- Error handling and timeout management

---

## Project Structure
```
LLM_QA_Project_YourName_MatricNo/
├── LLM_QA_CLI.py              # Command-line interface
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LLM_QA_hosted_webGUI_link.txt  # Deployment link
```

---

## Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- OpenAI API key (optional - can use mock mode)



## Usage

### Command Line Interface (CLI)

**Run with real API:**
```bash
export OPENAI_API_KEY=sk-your-key
python LLM_QA_CLI.py
```

**Run with mock responses (no API key needed):**
```bash
export USE_MOCK=true
python LLM_QA_CLI.py
```

**Example interaction:**
```
Enter your question: What is machine learning?
Processed Question: what is machine learning
Answer: Machine learning is a subset of artificial intelligence...
```

Type `quit` to exit.

---

### Web Interface (Streamlit)

**Run locally:**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

**Features:**
- Enter questions in text area
- View processed questions
- See AI-generated answers
- Browse last 5 questions in sidebar
- Toggle mock mode for testing
- Copy answers to clipboard

---

## Deployment

### Streamlit Cloud (Live Demo)

**Live URL:** [Your deployed URL will go here]

**How to deploy:**

1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Sign in with GitHub
4. Click "New app"
5. Select repository and `app.py`
6. Add `OPENAI_API_KEY` in secrets (optional)
7. Deploy

**Environment Variables (Streamlit Secrets):**
```toml
OPENAI_API_KEY = "sk-your-api-key"
```

---

## Technical Details

### Text Preprocessing
- Converts to lowercase
- Removes punctuation
- Tokenizes by whitespace
- Rejoins tokens into clean string

### LLM API Integration
- Model: `gpt-3.5-turbo`
- Max tokens: 200
- Temperature: 0.7
- Timeout: 30 seconds
- Error handling for rate limits, timeouts, invalid keys

### Prompt Template
```
System: You are a helpful assistant. Provide concise answers and cite sources when possible.
User: [preprocessed question]
```

---

## Dependencies

- `streamlit==1.29.0` - Web framework
- `openai==1.6.1` - OpenAI API client
- `python-dotenv==1.0.0` - Environment variable management

---

## Testing

### Test Checklist
- ✅ CLI runs without errors
- ✅ Questions are preprocessed correctly
- ✅ Mock mode works offline
- ✅ Real API returns answers
- ✅ Web UI displays correctly
- ✅ History updates properly
- ✅ Error handling works
- ✅ Deployed app is accessible

### Common Issues & Solutions

**"OPENAI_API_KEY not found"**
- Solution: Set environment variable or enable mock mode

**"Module not found"**
- Solution: Run `pip install -r requirements.txt`

**Streamlit won't start**
- Solution: Check if port 8501 is available, try `streamlit run app.py --server.port 8502`

---


### CLI Interface
```
============================================================
LLM Question & Answer System - CLI
============================================================
Enter your question: What is Python?
Processed Question: what is python
Answer: Python is a high-level programming language...
```



## License

This project is submitted as coursework for CSC415 at Covenant University.

---

## Contact

**Name:** Adesola Aderewa
**Matric No:** 22CG031811  
**Email:** aadesola.2200156@stu.cu.edu.ng
**GitHub:** https://github.com/aderewa03
---

## Acknowledgments

- OpenAI for GPT API
- Streamlit for web framework
- Course instructor and TAs

---

**Project Submission Date:** Friday, 21 November 2025, 12:00 PM WAT