
import os
import re
import json
import dill
from pypdf import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vertexai.preview.language_models import TextGenerationModel
from flask import Flask, render_template, request, jsonify, make_response



app = Flask(__name__)

DATA_PATH = 'sample_data'

def generate_reply(prompt):
    model = TextGenerationModel.from_pretrained('text-bison')
    response = model.predict(
        prompt,
        max_output_tokens=800,
        temperature=0,)
    return response.text

def get_prompt(file_name):
    path = os.path.join(DATA_PATH, file_name)
    with open(path) as f:
        prompt = f.read()
    return prompt

def load_data(file_name):
    path = os.path.join(DATA_PATH, file_name)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data
    
def parse_pdf(file):
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Fix newlines in the middle of sentences
        text = re.sub(r'(?<!\n\s)\n(?!\s\n)', ' ', text.strip())
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)

        output.append(text)

    return output

# helper functions to save and load vectorstore from file
def save_vectorstore_to_file(file_name, vectorstore):
    with open(file_name, 'wb') as f:
        dill.dump(vectorstore, f)

def load_vectorstore_from_file(file_name):
    with open(file_name, 'rb') as f:
        vectorstore = dill.load(f)
    return vectorstore


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/sentiment_analysis')
def sentiment_analysis():
    data = load_data('sentiment.json')
    return render_template('sentiment.html', data=data)


@app.route('/chat_itau')
def chat_itau():
    return render_template('chat_itau.html')


@app.route('/chat_pdf')
def chat_pdf():
    return render_template('chat_pdf.html')


@app.route('/itau_chat', methods=['POST'])
def itau_chat():
    query = request.get_json()['message']
    prompt = get_prompt('itau_prompt_pt.txt')
    context = get_prompt('itau_context.txt')
    full_prompt = prompt.format(context, query)
    reply = generate_reply(full_prompt)
    response = make_response(jsonify({'response': reply}))
    response.headers.set('Access-Control-Allow-Origin', '*')
    return response
 

@app.route('/pdf_chat', methods=['POST'])
def pdf_chat():
    query = request.get_json()['message']
    prompt = get_prompt('qa_prompt.txt')
    vectorstore = load_vectorstore_from_file(file_name='main.db')
    docs = vectorstore.similarity_search(query, k=4)
    docs = [doc.page_content for doc in docs]
    context = '\n'.join(docs)
    full_prompt = prompt.format(context, query)
    reply = generate_reply(full_prompt)
    response = make_response(jsonify({'response': reply}))
    return response


@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = file.filename
        file.save(os.path.join(os.getcwd(), filename))

        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 1500,
            chunk_overlap  = 200,
            length_function = len,
        )
        raw_text = parse_pdf(filename)
        text = text_splitter.split_text(raw_text)
        embeddings = GooglePalmEmbeddings()
        vectorstore = FAISS.from_texts(text, embeddings)
        save_vectorstore_to_file('main.db', vectorstore)
        return 'File saved successfully', 200


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the 'static' directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)