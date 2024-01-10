# FROM python:3.10.12
FROM python:3.10.12-slim-bullseye
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install --progress-bar off -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .
# EXPOSE 5000
# CMD python appmysql.py
# CMD flask --app app run
CMD ["flask", "--app", "app", "run", "--host=0.0.0.0"]
