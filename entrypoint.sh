#!/bin/bash

flask --app appstream run --host=0.0.0.0 &
streamlit run Chatbot.py &
wait -n
exit $?
