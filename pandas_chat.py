from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import os
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import format_document


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

class PandasChat:
    def __init__(self, sav_dir:str, batch_token:str):
        self.sav_dir = sav_dir
        self.index = batch_token
        self.db = FAISS(embedding_function=OpenAIEmbeddings())
        
        self.memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )
        self.loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    def chat(self, user_question):
        df = pd.read_excel(
            os.path.join(self.sav_dir, 'post_criteria_evaluation.xlsx')
        )

        retriever = self.db.as_retriever()
        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | CONDENSE_QUESTION_PROMPT
            | self.llm
            | StrOutputParser(),
        }

        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["standalone_question"],
        }
        # Now we construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"]),
            "question": itemgetter("question"),
        }
        # And finally, we do the part that returns the answers
        answer = {
            "answer": final_inputs | ANSWER_PROMPT | self.llm,
            "docs": itemgetter("docs"),
        }
        # And now we put it all together!
        final_chain = self.loaded_memory | standalone_question | retrieved_documents | answer
        inputs = {"question": user_question}
        result = final_chain.invoke(inputs)
        self.memory.save_context(inputs, {"answer": result["answer"].content})

        return result["answer"].content

    # def default_chat(self, user_question):
    #     df = pd.read_excel(
    #         os.path.join(self.sav_dir, "post_criteria_evaluation.xlsx")
    #     )
    #     df_dict = df.to_dict()
    #     llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    #     messages = [
    #         ("system", "You are a helpful assistant that helps to answer questions about a list of evaluation results from the candidate's resume. You will be provided context about the evaluation results. "),
    #         ("system", f"[Evaluation Results]{df_dict}[End of Evaluation Results]"),
    #         ("human", user_question),
    #     ]
    #     llm.invoke(messages)
        
    #     return messages['output']
