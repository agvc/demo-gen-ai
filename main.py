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

DATA_PATH = 'data'

def generate_reply(prompt):
    # Call the Palm API to get the text completion.
    model = TextGenerationModel.from_pretrained('text-bison')
    response = model.predict(
        prompt,
        max_output_tokens=800,
        temperature=0,)
    return response.text

def get_prompt(file_name):
    # Read the prompts from /data folder.
    path = os.path.join(DATA_PATH, file_name)
    with open(path) as f:
        prompt = f.read()
    return prompt

def load_data(file_name):
    # Read the sentiment cluster file from /data folder.
    path = os.path.join(DATA_PATH, file_name)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def parse_pdf(file):
    # Function to parse the pdf into a list of pages text.
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

# helper function to save vectorstore to file.
def save_vectorstore_to_file(file_name, vectorstore):
    with open(file_name, 'wb') as f:
        dill.dump(vectorstore, f)

# helper function to load vectorstore to file.
def load_vectorstore_from_file(file_name):
    with open(file_name, 'rb') as f:
        vectorstore = dill.load(f)
    return vectorstore


@app.route('/')
def home():
    data = load_data('sentiment.json')
    return render_template('sentiment.html', data=data)

@app.route('/sentiment_analysis')
def sentiment_analysis():
    data = load_data('sentiment.json')
    return render_template('sentiment.html', data=data)


@app.route('/chat_itau')
def chat_itau():
    return render_template('chat_itau.html')


@app.route('/chat_context')
def chat_context():
    return render_template('chat_context.html')


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


@app.route('/context_chat', methods=['POST'])
def context_chat():
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
            chunk_size = 1500,
            chunk_overlap  = 200,
            length_function = len,
        )
        file_ext = os.path.splitext(filename)[1]
        if file_ext == '.pdf':
            raw_text = parse_pdf(filename)
        elif file_ext == '.txt':
            with open(filename) as f:
                raw_text = f.read()
        text = text_splitter.split_text(raw_text)
        embeddings = GooglePalmEmbeddings()
        vectorstore = FAISS.from_texts(text, embeddings)
        save_vectorstore_to_file('main.db', vectorstore)
        return 'File saved successfully', 200


if __name__ == '__main__':
    print('To access the UI click on the link below:')

    # Set a proxy url to the local flask app.
    print(eval_js("google.colab.kernel.proxyPort(5000)"))
    app.run()