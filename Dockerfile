FROM python:3.10.12-slim-bullseye
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .
EXPOSE 5000 8501
CMD ./entrypoint.sh
