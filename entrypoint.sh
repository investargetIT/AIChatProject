#!/bin/bash

flask --app appBAAI run --host=0.0.0.0 &
streamlit run Chatbot.py &
wait -n
exit $?
