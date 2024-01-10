#!/bin/bash

docker run -d -p 5000:5000 -v $(pwd)/upload:/app/upload -v $(pwd)/sentence_transformers:/root/.cache/torch/sentence_transformers investarget/ai-chat-project flask --app appBAAI run --host=0.0.0.0

# docker run -it --rm -p 5000:5000 -v $(pwd)/upload:/app/upload -v $(pwd)/sentence_transformers:/root/.cache/torch/sentence_transformers investarget/ai-chat-project flask --app appBAAI run --host=0.0.0.0
