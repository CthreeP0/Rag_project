from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from llama_index.core.schema import Document
import uuid
import streamlit as st
import time
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field


class Candidate(BaseModel):
    """Information about a candidate from his/her resume."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    name: Optional[str] = Field(..., description="The name of the candidate")
    phone_number: Optional[str] = Field(
        ..., description="The phone number of the candidate"
    )
    email: Optional[str] = Field(
        ..., description="The email of the candidate"
    )
    local: Optional[str] = Field(
        ..., description="Is the candidate Malaysian(Yes or No)?"
    )
    expected_salary: Optional[str] = Field(
        ..., description="Candidate's expected salary in RM if known. (If the currency is Ringgit Malaysia, assign the numerical value or range values only Eg:'3000-3100'. If in other currency, assign alongside currency)"
    )
    current_location: Optional[List] = Field(
        ..., description="Candidate's current location if known. If the candidate does not mention the country, assign the country based on the state and city (return it in a python list containing dictionary format like this 'Country': '', 'State': '', 'City': '' )"
    )
    education_background: Optional[List] = Field(
        ..., description="Every single candidate's education background. (field of study, level (always expand to long forms), cgpa, university, Start Date, Year of Graduation (Year in 4-digits only, remove month). All in a python dict format."
    )
    professional_certificate: Optional[List] = Field(
        ..., description="Candidate's professional certificates if known"
    )
    skill_group: Optional[List] = Field(
        ..., description="Candidate's skill groups if known"
    )
    technology_programs_tool: Optional[List] = Field(
        ..., description="Technology (Tools, Program, System) that the candidate knows if known."
    )
    language: Optional[List] = Field(
        ..., description="Languages that the candidate knows"
    )
    previous_job_roles: Optional[List] = Field(
        ..., description="Every single one of the candidate's (job title, job company, Industries (strictly classify according to to The International Labour Organization), start date and end date (only assign date time format if available. Do not assign duration), job location, Job Duration (Years) (if not in years, convert to years)) (If duration is stated, update the job duration instead.) in a python dict format."
    )


def generate_random_id():
    random_id = uuid.uuid4()
    return str(random_id)


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

def extract_information(file_path):
    # Define a custom prompt to provide instructions and any additional context.
    # 1) You can add examples into the prompt template to improve extraction quality
    # 2) Introduce additional parameters to take context into account (e.g., include metadata
    #    about the document from which the text was extracted.)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm with 20 years experience in the recruiting industry. You will be provided with candidate's resume. "
                "Extract relevant candidate's information mentioned in the following candidate's resume together with their properties. "
                "If you do not know the value of an attribute asked to extract, "
                "1) Please provide an accurate answers, no guessing."
                "2) Please return 'N/A' only if the information is not mentioned."
                "3) The response should strictly follow the Python dictionary format."
                "4) No need to return any reasoning as this is only for extraction of information."
                "5) Extracted Properties of all Start date and End date: "
                "* if the month is not stated, assume that start/end date is in the middle of the year. "
                "* should never include english words such as 'months', 'years', 'days'. "
                "* Instead, dates should be dates converted to the following format: "
                "* date values assigned are strictly in Python datetime format "
                """Strict Format of either one: 
                    YYYY
                    YYYY-MM or YYYYMM
                    YYYY-MM-DD or YYYYMMDD
                6) Ensure that for any duration (year) calculation: 
                * Any end date that indicates "Present", refers to today's date, which is {current_date}. 
                * Do not assume the work experiences are continuous without breaks.
                * Method of duration calculation: Subtract the end date from start date to get the number of months. Finally sum up all relevant durations and convert to years. 
                * Triple check your calculations. ","""
            ),
            ("human", "{text}"),
        ]
    )

    #loader = PyPDFLoader(file_path, extract_images=True)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    document_objects = []

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.3)
    runnable = prompt | llm.with_structured_output(schema=Candidate)
    result = runnable.invoke({"text": documents,"current_date":datetime.now()})

    return result

# def extract_information(file_path):
#     # Schema
#     schema = {
#         "properties": {
#             "name": {"type": "string"},
#             "phone_number": {"type": "string"},
#             "email": {"type": "string"},
#             "local": {"type": "string"},
#             "last role": {"type": "string"},
#             "years of experience": {"type": "string"},
#             "education level": {"type": "string"},
#             "CGPA": {"type": "integer"},
#             "University": {"type": "string"},
#             "Education Background": {"type": "string"},
#             "Data Science Background": {"type": "string"},
#             "Relevant experience": {"type": "string"},
#         },
#         "required": ["name", "height"],
#     }

#     loader = PyPDFLoader(file_path, extract_images=True)
#     documents = loader.load()
#     document_objects = []

#     # Run chain
#     llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125",verbose=True)
#     chain = create_extraction_chain(schema, llm,verbose=True)
#     result = chain.run(documents)
#     metadata = {'page_label': '1', 'file_name': file_path}  

#     # Create Document object
#     document = Document(metadata=metadata,text=str(result))
#     document_objects.append(document)
#     return document_objects

import base64

def get_table_download_link():
    """Generates a link allowing the data in a given Pandas dataframe to be downloaded"""
    with open('results.xlsx', 'rb') as file:
        data = file.read()
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="results.xlsx">Download Excel file</a>'


def init_session(self, clear: bool =False):
    if not self.chat_inited or clear:
        st.session_state[self._session_key] = {}
        time.sleep(0.1)
        self.reset_history(self._chat_name)
