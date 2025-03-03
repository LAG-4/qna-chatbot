import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Import replicate for Claude-3.7-sonnet option
import replicate

load_dotenv()

# Setup environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# Define a prompt template that includes chat history
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's question taking into account the conversation history."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# Global dictionary to store chat histories per session
chat_histories = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in chat_histories:
        chat_histories[session] = ChatMessageHistory()
    return chat_histories[session]

def generate_response(prompt_text, backend, engine, api_key=None):
    if backend == "Groq":
        llm = ChatGroq(model_name=engine, max_tokens=4096)
        response = llm.invoke(prompt_text).content
    elif backend == "Ollama":
        llm = OllamaLLM(model=engine)
        response = llm.invoke(prompt_text)
    elif backend == "OpenAI":
        if not api_key:
            raise ValueError("API key required for OpenAI backend")
        openai_llm = ChatOpenAI(model=engine, api_key=api_key)
        response = openai_llm.invoke(prompt_text, max_tokens=10000).content
    elif backend == "Claude":
        # Use Replicate's run API for Anthropic Claude-3.7-Sonnet
        output = replicate.run(
            "anthropic/claude-3.7-sonnet",
            input={"prompt": prompt_text}
        )
        # If output is a list, join it to form a string response
        response = "".join(output) if isinstance(output, list) else output
    else:
        raise ValueError("Invalid backend specified")
    return response

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return "Welcome to the Chatbot API with Chat History!"

@app.route('/chat', methods=['POST'])
def chat():
    # Expecting a JSON payload
    data = request.get_json()
    question = data.get('question')
    backend = data.get('backend')  # "Groq", "Ollama", "OpenAI", or "claude-3.7-sonnet"
    engine = data.get('engine')
    session_id = data.get('session_id', "default_session")  # Session id for chat history

    if not question or not backend or not engine:
        return jsonify({"error": "Missing required parameters: question, backend, engine, and optionally session_id"}), 400

    try:
        # Retrieve chat history for the session
        history = get_session_history(session_id)
        # Format prompt with chat history and the current question
        prompt_text = chat_prompt.format(chat_history=history.messages, question=question)
        # Generate a response using the chosen backend
        response_text = generate_response(prompt_text, backend, engine, data.get('api_key'))
        # Update the session's chat history
        history.add_user_message(question)
        history.add_ai_message(response_text)
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
