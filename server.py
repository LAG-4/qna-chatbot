import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from dotenv import load_dotenv

# Import the required LangChain components and LLMs
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

def generate_response_ollama(question, engine, temperature, max_tokens):
    llm = OllamaLLM(model=engine, temperature=temperature, max_tokens=max_tokens)
    prompt_text = prompt.format(question=question)
    response = llm.invoke(prompt_text)
    return response

def generate_response_groq(question, engine, temperature, max_tokens):
    llm = ChatGroq(model_name=engine, temperature=temperature, max_tokens=max_tokens)
    prompt_text = prompt.format(question=question)
    response = llm.invoke(prompt_text).content
    return response

def generate_response_openai(question, api_key, model_name, temperature, max_tokens):
    openai_llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    prompt_text = prompt.format(question=question)
    response = openai_llm.invoke(prompt_text)
    return response.content

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/chat', methods=['POST'])
def chat():
    # Expecting a JSON payload
    data = request.get_json()

    question = data.get('question')
    backend = data.get('backend')  # should be "Groq", "Ollama", or "OpenAI"
    engine = data.get('engine')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 150)

    if not question or not backend or not engine:
        return jsonify({"error": "Missing required parameters: question, backend, and engine are required"}), 400

    try:
        if backend == "Groq":
            response_text = generate_response_groq(question, engine, temperature, max_tokens)
        elif backend == "Ollama":
            response_text = generate_response_ollama(question, engine, temperature, max_tokens)
        elif backend == "OpenAI":
            api_key = data.get('api_key')
            if not api_key:
                return jsonify({"error": "API key required for OpenAI backend"}), 400
            response_text = generate_response_openai(question, api_key, engine, temperature, max_tokens)
        else:
            return jsonify({"error": "Invalid backend specified"}), 400

        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
