from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


class DefaultChat:
    def chat(self, user_question):
        raw_documents = TextLoader('README.md').load()

        text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
        db = FAISS.from_documents(documents, OpenAIEmbeddings())

        retriever = db.as_retriever()
        template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

        Question: {question} 

        Context: {context} 

        Answer:
        """

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = rag_chain.invoke(user_question)

        return result
