
import os
import string
import sys
from openai import OpenAI
import time

# ============= PROMPT TEMPLATES =============

SYSTEM_PROMPT = """You are a helpful assistant. Provide concise answers and cite sources when possible."""

# Basic user prompt (currently used)
USER_PROMPT_BASIC = "{processed_question}"

# Enhanced user prompt (more structured)
USER_PROMPT_ENHANCED = """Please answer the following question concisely:

Question: {processed_question}

Provide a clear, factual answer. If applicable, cite reliable sources."""

# ============================================

def preprocess_question(question: str) -> str:
    """
    Preprocess the input question:
    - Convert to lowercase
    - Tokenize by whitespace
    - Remove punctuation
    """
    # Convert to lowercase
    question_lower = question.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    question_no_punct = question_lower.translate(translator)
    
    # Tokenize and rejoin (removes extra spaces)
    tokens = question_no_punct.split()
    processed = ' '.join(tokens)
    
    return processed

def build_prompt(processed_question: str, use_enhanced: bool = False) -> list:
    """
    Build the prompt messages for the LLM API using templates.
    
    Args:
        processed_question: The preprocessed user question
        use_enhanced: If True, uses enhanced prompt template
    
    Returns:
        List of message dictionaries for the API
    """
    # Choose prompt template
    if use_enhanced:
        user_content = USER_PROMPT_ENHANCED.format(processed_question=processed_question)
    else:
        user_content = USER_PROMPT_BASIC.format(processed_question=processed_question)
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    return messages

def call_llm_api(messages: list, use_mock: bool = False) -> str:
    """
    Send prompt to OpenAI API and return response.
    Includes error handling, timeout, and retry logic.
    """
    if use_mock:
        return mock_llm_response(messages[-1]["content"])
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it:\n"
            "export OPENAI_API_KEY='your-api-key-here'"
        )
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Make API call with timeout
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,  # Limit response length
            temperature=0.7,
            timeout=30.0  # 30 second timeout
        )
        
        # Extract answer
        answer = response.choices[0].message.content.strip()
        return answer
        
    except Exception as e:
        error_type = type(e).__name__
        
        # Handle specific errors
        if "timeout" in str(e).lower():
            return f"Error: Request timed out. Please try again."
        elif "rate_limit" in str(e).lower():
            return f"Error: Rate limit exceeded. Please wait a moment and try again."
        elif "authentication" in str(e).lower():
            return f"Error: Invalid API key. Please check your OPENAI_API_KEY."
        else:
            return f"Error ({error_type}): {str(e)}"

def mock_llm_response(question: str) -> str:
    """
    Fallback mock function for offline testing.
    Returns canned responses based on keywords.
    """
    question_lower = question.lower()
    
    if "capital" in question_lower or "city" in question_lower:
        return "Mock Response: The capital varies by country. For example, the capital of France is Paris."
    elif "python" in question_lower or "code" in question_lower:
        return "Mock Response: Python is a high-level programming language known for readability and versatility."
    elif "weather" in question_lower or "temperature" in question_lower:
        return "Mock Response: I cannot access real-time weather data. Please check a weather service."
    else:
        return f"Mock Response: This is a simulated answer to your question about '{question[:50]}...'"

def main():
    """
    Main function to run the CLI Q&A system.
    """
    print("=" * 60)
    print("LLM Question & Answer System - CLI")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    
    # Check if we should use mock mode (for testing without API key)
    use_mock = os.getenv("USE_MOCK", "false").lower() == "true"
    if use_mock:
        print("[MOCK MODE: Using simulated responses]\n")
    
    while True:
        # Get user input
        question = input("Enter your question: ").strip()
        
        # Exit condition
        if question.lower() == 'quit':
            print("\nThank you for using the LLM Q&A System!")
            break
        
        # Skip empty questions
        if not question:
            print("Please enter a valid question.\n")
            continue
        
        print("\nProcessing your question...\n")
        
        # Step 1: Preprocess
        processed_question = preprocess_question(question)
        print(f"Processed Question: {processed_question}")
        
        # Step 2: Build prompt
        messages = build_prompt(processed_question)
        
        # Step 3: Call LLM API
        print("\nFetching answer from LLM...\n")
        answer = call_llm_api(messages, use_mock=use_mock)
        
        # Step 4: Display answer
        print("-" * 60)
        print("ANSWER:")
        print(answer)
        print("-" * 60)
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
        sys.exit(0)