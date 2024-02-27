#!pip install langchain
#!pip install sentence-transformers
#!pip install faiss-gpu
#!pip install pypdf

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

loader = PyPDFLoader('Ang Teik Hun Resume.pdf')
#loader = TextLoader("./sample_text_file.txt")

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)


turbo_llm = ChatOpenAI(
    temperature=0.3,
    model_name='gpt-3.5-turbo-0125'
)

from langchain.chains import RetrievalQA
from langchain.schema import retriever

retriever = db.as_retriever()
chain = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff", retriever=retriever)

query="When was this candidate graduated?"
out = chain.invoke(query)
print(out['result'])

