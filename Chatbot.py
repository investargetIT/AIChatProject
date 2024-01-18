from openai import OpenAI
import streamlit as st
from config import OPENAI_API_BASE, OPENAI_API_KEY

st.write('<div style="display: flex;align-items: center;"><img src="http://peidibrand.com/assets/images/logo.png" /></div>', unsafe_allow_html=True)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

def update_system_prompt():
    st.session_state.messages[0]['content'] = st.session_state.system_prompt

st.sidebar.radio(
    "模型",
    ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
    key="openai_model"
)

st.sidebar.text_area(
    "系统提示词",
    "You are a helpful assistant.",
    200,
    key="system_prompt",
    on_change=update_system_prompt,
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-1106"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": st.session_state.system_prompt}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
