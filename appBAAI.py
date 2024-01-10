import datetime
import json
import os
import random
import string
import traceback
# from langchain_community.utilities.sql_database import SQLDatabase
# from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain
import openai
import requests
from flask import Flask
from flask import request
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceBgeEmbeddings
from urllib.parse import quote_plus as urlquote
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from werkzeug.utils import secure_filename

# GRANT ALL PRIVILEGES ON *.* to 'root'@'223.166.124.216' identified by 'Investarget@2017' WITH GRANT OPTION;
# flush privileges;
os.environ["OPENAI_API_BASE"] = 'http://8.219.158.94:4056/v1'
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
        OPENAI_API_KEY = aidata['key']

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

        # ZILLIZ_ENDPOINT = request.form.get('zilliz_url')
        ZILLIZ_ENDPOINT = 'https://in03-d3b6ba8bbd0dfd0.api.gcp-us-west1.zillizcloud.com'
        # ZILLIZ_token = request.form.get('zilliz_key')
        ZILLIZ_token = '8e2b989c1a4c170e5c47b5324c0630d23c3085654923a654e3df081dbeae9c31c95923af0ec0d2bc4cf9ce5bb9d6fca43051c37d'
        zilliz_collection_name = request.form.get('zilliz_collection_name')
        OPENAI_API_KEY = request.form.get('open_ai_key')

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        # embeddings = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                              model_kwargs=model_kwargs,
                                              encode_kwargs=encode_kwargs)
        loader = PDFMinerLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        logdata(file_key)
        Milvus.from_documents(docs, embeddings, collection_name=zilliz_collection_name, connection_args={"uri": ZILLIZ_ENDPOINT, "token": ZILLIZ_token})
        return {'success': True, 'result': file_key, 'errmsg': None}
    except Exception:
        msg = {'success': False, 'result': None, 'errmsg': traceback.format_exc()}
        print(msg['errmsg'])
        return msg


@app.route('/zillizchat/', methods=['POST'])
def chatgptWithZillizCloud():
    try:

        # ZILLIZ_ENDPOINT = request.json.get('zilliz_url')
        ZILLIZ_ENDPOINT = 'https://in03-d3b6ba8bbd0dfd0.api.gcp-us-west1.zillizcloud.com'
        # ZILLIZ_token = request.json.get('zilliz_key')
        ZILLIZ_token = '8e2b989c1a4c170e5c47b5324c0630d23c3085654923a654e3df081dbeae9c31c95923af0ec0d2bc4cf9ce5bb9d6fca43051c37d'
        zilliz_collection_name = request.json.get('zilliz_collection_name')
        chat_model = request.json.get('chat_model')
        OPENAI_API_KEY = request.json.get('open_ai_key')
        question = request.json.get('question')
        list_chat_history = request.json.get('chat_history', [])
        logdata(list_chat_history)
        tuple_chat_history = []
        for history in list_chat_history:
            tuple_chat_history.append(tuple(history))
        logdata(tuple_chat_history)
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        # embeddings = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                              model_kwargs=model_kwargs,
                                              encode_kwargs=encode_kwargs)
        openai_ojb = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name=chat_model)
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
        OPENAI_API_KEY = request.json.get('open_ai_key')
        question = request.json.get('question')
        file_key = request.json.get('file_key')
        file_path = os.path.join(UPLOAD_FOLDER, file_key)
        loader = PDFMinerLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
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




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
