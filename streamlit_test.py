import streamlit as st
from default_chat import DefaultChat
from streamlit_pills import pills
import os
from dotenv import load_dotenv
import pandas as pd


load_dotenv(".env")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"FYP-Goo"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get('LANGCHAIN_API_KEY')

st.title("title bar : Chat with docs")
st.info("Info bar", icon="📃")

df = pd.read_excel('results.xlsx',index_col=0)

st.write(df)


predefined_prompt_selected = pills("Q&A", ["What is the purpose of the Resume Parser tool?", "What are the expected outputs for the extracted results?", "What is the format for the evaluating criteria details that users should follow?"], ["🍀", "🎈", "🌈"],clearable=True,index = None)

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    index = DefaultChat()
    st.session_state.chat_engine = index

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": f"Ask me a question about Resume Parser!"}
    ]

if predefined_prompt_selected:
    prompt = predefined_prompt_selected
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = st.session_state.chat_engine.chat(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response}) # Add response to message history

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response}) # Add response to message history



