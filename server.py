import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Setup environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# Define the common prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Answer the users questions"),
        ("user", "Question:{question}")
    ]
)

def generate_response_ollama(question, engine):
    llm = OllamaLLM(model=engine)
    prompt_text = prompt.format(question=question)
    response = llm.invoke(prompt_text)
    return response

def generate_response_groq(question, engine):
    # Initialize ChatGroq with a default max_tokens of 4096
    llm = ChatGroq(model_name=engine, max_tokens=4096)
    prompt_text = prompt.format(question=question)
    response = llm.invoke(prompt_text).content
    return response


def generate_response_openai(question, api_key, model_name):
    openai_llm = ChatOpenAI(model=model_name, api_key=api_key)
    prompt_text = prompt.format(question=question)
    # Increase or remove max_tokens as needed
    response = openai_llm.invoke(prompt_text, max_tokens=10000).content
    return response


# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return "Welcome to the Chatbot API!"

@app.route('/chat', methods=['POST'])
def chat():
    # Expecting a JSON payload
    data = request.get_json()

    question = data.get('question')
    backend = data.get('backend')  # should be "Groq", "Ollama", or "OpenAI"
    engine = data.get('engine')

    if not question or not backend or not engine:
        return jsonify({"error": "Missing required parameters: question, backend, and engine are required"}), 400

    try:
        if backend == "Groq":
            response_text = generate_response_groq(question, engine)
        elif backend == "Ollama":
            response_text = generate_response_ollama(question, engine)
        elif backend == "OpenAI":
            api_key = data.get('api_key')
            if not api_key:
                return jsonify({"error": "API key required for OpenAI backend"}), 400
            response_text = generate_response_openai(question, api_key, engine)
        else:
            return jsonify({"error": "Invalid backend specified"}), 400

        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run() # Debug mode is off.