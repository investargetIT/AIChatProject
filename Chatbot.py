from openai import OpenAI
import streamlit as st

openai_api_key = "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"
openai_api_base = 'http://8.219.158.94:4056/v1'

st.write('<div style="display: flex;align-items: center;"><img src="http://peidibrand.com/assets/images/logo.png" /><span style="margin-left: 10px;font-size: 32px;font-weight: bold;">Peidi 机器人</span></div>', unsafe_allow_html=True)
st.caption("🚀 Powered by Streamlit")

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

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
