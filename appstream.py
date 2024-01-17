import datetime
import json
import os
import random
import string
import threading
import traceback
from typing import Any
import requests
from flask import Flask, Response
from flask import request
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import  HuggingFaceBgeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain_core.callbacks import StreamingStdOutCallbackHandler, CallbackManager
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from werkzeug.utils import secure_filename

from config import ZILLIZ_ENDPOINT, ZILLIZ_TOKEN, OPENAI_API_KEY, ZILLIZ_COLLECTION_NAME, OPENAI_CHAT_MODEL, \
    Embedding_model_name, Embedding_model_kwargs, OPENAI_API_BASE, chat_template

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
        # OPENAI_API_KEY = aidata['key']
        headers = {
            'Content-Type': "application/json",
            'Authorization': "Bearer {}".format(OPENAI_API_KEY)
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
@app.route('/embedzilliz/', methods=['POST'])
def embeddingFileAndUploadToZillizCloud():
    try:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filetype = filename.split('.')[-1]
            temp_key = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ''.join(
                random.sample(string.ascii_lowercase, 6))
            file_key = '%s.%s' % (temp_key, filetype)
            file_path = os.path.join(UPLOAD_FOLDER, file_key)
            file.save(file_path)
        else:
            return {'success': False, 'result': None, 'errmsg': '文件不合格'}

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceBgeEmbeddings(model_name=Embedding_model_name,
                                              model_kwargs=Embedding_model_kwargs,
                                              encode_kwargs=encode_kwargs)
        loader = PDFMinerLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        logdata(file_key)
        Milvus.from_documents(docs, embeddings, collection_name=ZILLIZ_COLLECTION_NAME, connection_args={"uri": ZILLIZ_ENDPOINT, "token": ZILLIZ_TOKEN})
        return {'success': True, 'result': file_key, 'errmsg': None}
    except Exception:
        msg = {'success': False, 'result': None, 'errmsg': traceback.format_exc()}
        logexcption(msg['errmsg'])
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

@app.route('/streamchat/', methods=['POST'])
def chat():
    try:
        # input_text = '狗粮、便宜、健康'
        input_text = request.json.get('question')
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
