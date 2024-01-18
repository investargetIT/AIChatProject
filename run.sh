#!/bin/bash

# For production
docker run -d -p 5000:5000 -p 8501:8501 -v $(pwd)/upload:/app/upload -v $(pwd)/Chatbot.py:/app/Chatbot.py -v $(pwd)/appstream.py:/app/appstream.py -v $(pwd)/config.py:/app/config.py -v $(pwd)/sentence_transformers:/root/.cache/torch/sentence_transformers investarget/ai-chat-project

# For development
# docker run -it --rm -p 5000:5000 -p 8501:8501 -v $(pwd):/app -v $(pwd)/sentence_transformers:/root/.cache/torch/sentence_transformers investarget/ai-chat-project
