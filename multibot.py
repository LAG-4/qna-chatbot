import os 
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
import openai
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

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

st.title("Enhanced Q&A Chatbot With OpenAI")
st.sidebar.title("Settings")
backend = st.sidebar.selectbox("Select if you want to use Ollama (local), Groq, or OpenAI", ["Groq", "Ollama", "OpenAI"])

if backend == "Groq":
    engine = st.sidebar.selectbox("Select your Groq model", ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"])
elif backend == "Ollama":
    engine = st.sidebar.selectbox("Select your OLLAMA model", ["deepseek-coder:latest", "deepseek-r1:latest"])
elif backend == "OpenAI":
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    engine = st.sidebar.selectbox("Select your OpenAI model", ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Go ahead and ask a question")
user_input = st.text_input("You:")

if user_input:
    if backend == "Groq":
        response = generate_response_groq(user_input, engine, temperature, max_tokens)
        st.write(response)
    elif backend == "Ollama":
        response = generate_response_ollama(user_input, engine, temperature, max_tokens)
        st.write(response)
    elif backend == "OpenAI":
        response = generate_response_openai(user_input, api_key, engine, temperature, max_tokens)
        st.write(response)
else:
    st.write("Please provide a query")
