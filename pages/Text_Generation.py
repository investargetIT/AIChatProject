import streamlit as st
import requests

options = {
    "TONGYI": "通义千问",
    # "BAICHUAN": "百川大模型",
    "OPENAI": "GPT",
}
st.image('http://peidibrand.com/assets/images/logo.png')
st.text_area(
    '关键字',
    '爵宴系列狗粮、唤醒狗狗的食欲、爵宴纯粹鹿肺块、新西兰原装进口、原切鹿肺、尽量每句话都有使用emoji表情、单句不要超过20字、300字左右',
    key='key_word'
)
st.radio(
    "文本类型",
    ["文案向", "内容向", "活动向", "数据向"],
    key="word_type"
)
st.radio(
    "模型",
    [
        "TONGYI",
        # "BAICHUAN",
        "OPENAI"
    ],
    format_func=lambda a: options[a],
    key="llm"
)
if st.button("生成", type="primary"):
    message_placeholder = st.empty()
    full_response = ""
    r = requests.post('http://127.0.0.1:5000/streamchat/', json={'keyWord': st.session_state.key_word, 'wordType': st.session_state.word_type, 'ai_type': st.session_state.llm}, stream=True)
    for chunk in r.iter_content(decode_unicode=True):
        full_response += chunk
        message_placeholder.text(full_response + "▌")
    message_placeholder.text(full_response)
