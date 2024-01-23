import streamlit as st
import requests

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
if st.button("生成", type="primary"):
    message_placeholder = st.empty()
    full_response = ""
    r = requests.post('http://127.0.0.1:5000/streamchat/', json={'keyWord': st.session_state.key_word, 'wordType': st.session_state.word_type}, stream=True)
    for chunk in r.iter_content(decode_unicode=True):
        full_response += chunk
        message_placeholder.write(full_response + "▌")
    message_placeholder.write(full_response)
