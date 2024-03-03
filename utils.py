from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from llama_index.core.schema import Document
import uuid

def generate_random_id():
    random_id = uuid.uuid4()
    return str(random_id)

def extract_information(file_path):
    # Schema
    schema = {
        "properties": {
            "name": {"type": "string"},
            "phone_number": {"type": "string"},
            "email": {"type": "string"},
            "local": {"type": "string"},
            "last role": {"type": "string"},
            "years of experience": {"type": "string"},
            "education level": {"type": "string"},
            "CGPA": {"type": "integer"},
            "University": {"type": "string"},
            "Education Background": {"type": "string"},
            "Data Science Background": {"type": "string"},
            "Relevant experience": {"type": "string"},
        },
        "required": ["name", "height"],
    }

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    document_objects = []

    # Run chain
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125",verbose=True)
    chain = create_extraction_chain(schema, llm,verbose=True)
    result = chain.run(documents)
    metadata = {'page_label': '1', 'file_name': file_path}  


    # Create Document object
    document = Document(metadata=metadata,text=str(result))
    document_objects.append(document)
    return document_objects
