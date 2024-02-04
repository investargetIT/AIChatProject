#!/bin/bash

flask --app appstream run --host=0.0.0.0 &
streamlit run pages/Text_Generation.py &
wait -n
exit $?
