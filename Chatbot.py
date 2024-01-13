import streamlit as st
import requests

openai_api_key = "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"

st.title("💬 聊天机器人")
st.caption("🚀 Powered by Streamlit")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    r = requests.post('http://39.107.14.53:5000/zillizchat/', json={
            "zilliz_collection_name": "sentenceembeddingdata",
            "chat_model": "gpt-3.5-turbo-1106",
            "open_ai_key": openai_api_key,
            "chat_history": [],
            "question": prompt
        })
    res = r.json()
    print(res)
    msg = res['result']['answer']

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
