from dotenv import load_dotenv
from flask import Flask, request, render_template, send_file, jsonify, make_response
import os
from uuid import uuid4
import secrets
import string

app = Flask(__name__)
UPLOAD_FOLDER = "upload"
DOWNLOAD_FOLDER = "download"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER


load_dotenv(".env")
openai_api_key = os.environ.get('OPENAI_API_KEY')
serp_api_key = os.environ.get('SERPER_API_KEY')


unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get('LANGCHAIN_API_KEY')

@app.route('/', methods=['GET', 'POST'])
def index():
    batch_token = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))

    return render_template('main.html', batch_token=batch_token)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)