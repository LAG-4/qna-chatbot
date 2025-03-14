import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

### Langsmith Tracking

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with OpenAI"

prompt = ChatPromptTemplate(
    [
        ("system","You are a helpful assistant. Please respond to ther user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    llm = ChatOpenAI(model=llm, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    prompt_text = prompt.format(question=question)
    response = llm.invoke(prompt_text)
    answer = response.content
    return answer

st.title("Enhanced Q&A Chatbot With OpenAI")

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your OpenAI API key",type="password")

llm=st.sidebar.selectbox("Select your OpenAI model",["gpt-4o","gpt-4o-mini","o1-mini","o3-mini"])

temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

st.write("Go ahead and ask a question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)

else:
    st.write("Please provide a query")
