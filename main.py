
import streamlit as st
from backend.core2 import run_llm

st.header("Langchain Documentation Helper")

if ("chat_history" not in st.session_state):
    st.session_state['chat_history'] = []

prompt = st.chat_input("Enter your message here")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    with st.spinner("Generating response.."):
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = run_llm(prompt, st.session_state['chat_history'])
            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
            st.markdown(response['answer'])

