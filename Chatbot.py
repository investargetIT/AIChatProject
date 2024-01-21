from openai import OpenAI
import streamlit as st
import joblib
import time
import os
from config import OPENAI_API_BASE, OPENAI_API_KEY

new_chat_id = f'{time.time()}'

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

def update_system_prompt():
    st.session_state.messages[0]['content'] = st.session_state.system_prompt

# Create a data/ folder if it doesn't already exist
try:
    os.mkdir('data/')
except:
    # data/ folder already exists
    pass

# Load past chats (if available)
try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}

st.image('http://peidibrand.com/assets/images/logo.png')

with st.sidebar:
    st.radio(
        "模型",
        ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
        key="openai_model"
    )

    st.text_area(
        "系统提示词",
        "You are a helpful assistant.",
        200,
        key="system_prompt",
        on_change=update_system_prompt,
    )
    
    st.session_state.chat_id = st.selectbox(
        label='聊天记录',
        options=[new_chat_id] + list(past_chats.keys()),
        format_func=lambda x: past_chats.get(x, 'New Chat'),
        placeholder='_',
    )
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-1106"

# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(
        f'data/{st.session_state.chat_id}-st_messages'
    )
    # print('old cache')
except:
    st.session_state.messages = [{"role": "system", "content": st.session_state.system_prompt}]
    # print('new_cache made')

# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "system", "content": st.session_state.system_prompt}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Save this as a chat for later
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')

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

    joblib.dump(
        st.session_state.messages,
        f'data/{st.session_state.chat_id}-st_messages',
    )
