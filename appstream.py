import datetime
import json
import os
import random
import string
import threading
import traceback
from typing import Any

import openai
import requests
from flask import Flask, Response
from flask import request
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import  HuggingFaceBgeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.callbacks import StreamingStdOutCallbackHandler, CallbackManager
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from werkzeug.utils import secure_filename

from config import OPENAI_API_BASE, chat_template, OPENAI_API_KEY, OPENAI_CHAT_MODEL

os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

app = Flask(__name__)
def logexcption(msg=None):
    now = datetime.datetime.now()
    filepath = 'excptionlog' + '/' + now.strftime('%Y-%m-%d')
    f = open(filepath, 'a')
    f.writelines(now.strftime('%H:%M:%S')+'\n'+ traceback.format_exc() + msg + '\n\n')
    f.close()

def logdata(data=None):
    now = datetime.datetime.now()
    filepath = 'excptionlog' + '/' + now.strftime('%Y-%m-%d')
    f = open(filepath, 'a')
    f.writelines(now.strftime('%H:%M:%S') + '\n' + json.dumps(data) + '\n\n')
    f.close()

@app.route('/ai/', methods=['POST'])
def getOpenAiChatResponse():
    try:
        chatdata = request.json['chatdata']
        aidata = request.json['aidata']
        OPENAI_URL = aidata['url']
        open_ai_key = aidata['key']
        headers = {
            'Content-Type': "application/json",
            'Authorization': "Bearer {}".format(open_ai_key)
        }
        res = requests.post(OPENAI_URL, data=json.dumps(chatdata), headers=headers).content.decode()
        return {'success': True, 'result': res, 'errmsg': None}
    except Exception:
        msg = {'success': False, 'result': None, 'errmsg': traceback.format_exc()}
        return msg


ALLOWED_EXTENSIONS = ['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']
# UPLOAD_FOLDER = '/var/www/AIChatProject/upload'
# UPLOAD_FOLDER = r'C:\Users\wjk13\Desktop\hokong'
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def get_chat_history(inputs) -> str:
    return inputs


@app.route('/zillizchat/', methods=['POST'])
def chatgptWithZillizCloud():
    try:

        # ZILLIZ_ENDPOINT = request.json.get('zilliz_url')
        ZILLIZ_ENDPOINT = 'https://in03-d3b6ba8bbd0dfd0.api.gcp-us-west1.zillizcloud.com'
        # ZILLIZ_token = request.json.get('zilliz_key')
        ZILLIZ_token = '8e2b989c1a4c170e5c47b5324c0630d23c3085654923a654e3df081dbeae9c31c95923af0ec0d2bc4cf9ce5bb9d6fca43051c37d'
        zilliz_collection_name = request.json.get('zilliz_collection_name')
        chat_model = request.json.get('chat_model')
        open_ai_key = request.json.get('open_ai_key')
        question = request.json.get('question')
        list_chat_history = request.json.get('chat_history', [])
        logdata(list_chat_history)
        tuple_chat_history = []
        for history in list_chat_history:
            tuple_chat_history.append(tuple(history))
        logdata(tuple_chat_history)
        os.environ["OPENAI_API_KEY"] = open_ai_key
        # embeddings = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                              model_kwargs=model_kwargs,
                                              encode_kwargs=encode_kwargs)
        openai_ojb = ChatOpenAI(temperature=0, openai_api_key=open_ai_key, model_name=chat_model)
        vector_db = Milvus(embeddings, zilliz_collection_name, {"uri": ZILLIZ_ENDPOINT, "token": ZILLIZ_token})
        chain = ConversationalRetrievalChain.from_llm(openai_ojb, vector_db.as_retriever())
        tuple_chat_history.reverse()
        result = chain({
            'question': question,  # 传入问题
            "chat_history": tuple_chat_history
        })
        return {'success': True, 'result': result, 'errmsg': None, 'reset': False}
    except openai.error.InvalidRequestError:
        msg = {'success': False, 'result': None, 'errmsg': traceback.format_exc(), 'reset': True}
        print(traceback.format_exc())
        return msg
    except Exception:
        msg = {'success': False, 'result': None, 'errmsg': traceback.format_exc(), 'reset': False}
        print(traceback.format_exc())
        return msg

@app.route('/zillizchat/', methods=['POST'])
def chatgptWithZillizCloud():
    try:

        ZILLIZ_ENDPOINT = request.json.get('zilliz_url')
        ZILLIZ_token = request.json.get('zilliz_key')
        zilliz_collection_name = request.json.get('zilliz_collection_name')
        chat_model = request.json.get('chat_model')
        open_ai_key = request.json.get('open_ai_key')
        question = request.json.get('question')
        list_chat_history = request.json.get('chat_history', [])
        logdata(list_chat_history)
        tuple_chat_history = []
        for history in list_chat_history:
            tuple_chat_history.append(tuple(history))
        logdata(tuple_chat_history)
        os.environ["OPENAI_API_KEY"] = open_ai_key
        embeddings = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
        openai_ojb = ChatOpenAI(temperature=0, openai_api_key=open_ai_key, model_name=chat_model)
        vector_db = Milvus(embeddings, zilliz_collection_name, {"uri": ZILLIZ_ENDPOINT, "token": ZILLIZ_token})
        chain = ConversationalRetrievalChain.from_llm(openai_ojb, vector_db.as_retriever())
        tuple_chat_history.reverse()
        result = chain({
            'question': question,  # 传入问题
            "chat_history": tuple_chat_history
        })
        return {'success': True, 'result': result, 'errmsg': None, 'reset': False}
    except openai.error.InvalidRequestError:
        msg = {'success': False, 'result': None, 'errmsg': traceback.format_exc(), 'reset': True}
        print(traceback.format_exc())
        return msg
    except Exception:
        msg = {'success': False, 'result': None, 'errmsg': traceback.format_exc(), 'reset': False}
        print(traceback.format_exc())
        return msg


@app.route('/pdfchat/', methods=['POST'])
def chatgptWithPDFFile():
    try:

        chat_model = request.json.get('chat_model')
        open_ai_key = request.json.get('open_ai_key')
        question = request.json.get('question')
        file_key = request.json.get('file_key')
        file_path = os.path.join(UPLOAD_FOLDER, file_key)
        loader = PDFMinerLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        os.environ["OPENAI_API_KEY"] = open_ai_key
        openai_ojb = ChatOpenAI(temperature=0, model_name=chat_model)
        chain = load_qa_chain(openai_ojb, chain_type="stuff")

        result = chain.run(input_documents=docs, question=question)
        return {'success': True, 'result': result, 'errmsg': None, 'reset': False}
    except openai.error.InvalidRequestError:
        msg = {'success': False, 'result': None, 'errmsg': traceback.format_exc(), 'reset': True}
        print(traceback.format_exc())
        return msg
    except Exception:
        msg = {'success': False, 'result': None, 'errmsg': traceback.format_exc(), 'reset': False}
        print(traceback.format_exc())
        return msg




class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.tokens = []
        # 记得结束后这里置true
        self.finish = False

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="")
        self.tokens.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.finish = 1

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        print(str(error))
        self.tokens.append(str(error))

    def generate_tokens(self):
        while not self.finish or self.tokens:
            if self.tokens:
                data = self.tokens.pop(0)
                yield data
            else:
                pass

def chat_bot(input_text, chat_history):
    handler = ChainStreamHandler()
    llm = ChatOpenAI(temperature=0, streaming=True, openai_api_key=OPENAI_API_KEY, model_name=OPENAI_CHAT_MODEL, callback_manager=CallbackManager([handler]))
    chat_prompt = ChatPromptTemplate.from_template(template=chat_template)
    # memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=200)
    # for history in chat_history:
    #     memory.save_context({"input": history[0]}, {"output": history[1]})
    # qa = LLMChain(llm=llm, prompt=chat_prompt, memory=memory, verbose=True)
    qa = LLMChain(llm=llm, prompt=chat_prompt)
    thread = threading.Thread(target=async_run, args=(qa, input_text))
    thread.start()
    return handler.generate_tokens()

def async_run(qa, input_text):
    qa.run(input=input_text)

@app.route('/streamchat/', methods=['GET'])
def chat():
    try:
        input_text = '猫粮、耐存储、健康'
        # input_text = request.json.get('question')
        # list_chat_history = request.json.get('chat_history', [])
        tuple_chat_history = []
        # for history in list_chat_history:
        #     tuple_chat_history.append(tuple(history))
        return Response(chat_bot(input_text, tuple_chat_history), mimetype="text/event-stream")
    except Exception:
        msg = {'success': False, 'result': None, 'errmsg': traceback.format_exc()}
        logexcption(msg['errmsg'])
        return msg


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
