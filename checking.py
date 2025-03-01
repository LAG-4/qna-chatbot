import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# Set up Streamlit interface
st.title("Chatbot with History")
st.write("Chat with the assistant while keeping track of conversation history.")


if api_key:
    # Initialize the Chat LLM with the provided API key
    llm = ChatGroq(groq_api_key=api_key, model_name="deepseek-r1-distill-llama-70b")

    # Input session ID for chat history management
    session_id = st.text_input("Session ID", value="default_session")

    # Create a dictionary in session state to hold chat histories
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Function to get or create the chat history for a given session
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # Define a simple chat prompt that uses the chat history
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Provide a concise and helpful answer."
    )
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # User input for the chat
    user_input = st.text_input("Your question:")

    if user_input:
        # Retrieve the chat history for the current session
        session_history = get_session_history(session_id)

        # Format the prompt using the chat history and latest user input
        prompt = chat_prompt.format(
            chat_history=session_history.messages, 
            input=user_input
        )

        # Call the LLM using the formatted prompt (which is now a string)
        response = llm.invoke(prompt)
        # If the response is a dictionary, try to extract the answer
        answer = response.get("answer", "") if isinstance(response, dict) else response

        # Display the answer from the assistant
        st.write("Assistant:", answer)

        # Update chat history with both the user input and the assistant's response
        session_history.add_user_message(user_input)
        session_history.add_ai_message(answer)