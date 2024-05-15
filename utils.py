from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
import uuid
import streamlit as st
import time
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import base64
from models import Candidate,Job
import pandas as pd



def extract_information(file_path,job_title):
    # Define a custom prompt to provide instructions and any additional context.
    # 1) You can add examples into the prompt template to improve extraction quality
    # 2) Introduce additional parameters to take context into account (e.g., include metadata
    #    about the document from which the text was extracted.)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert extraction algorithm with 20 years experience in the recruiting industry. You will be provided with candidate's resume.
                [Instruction] Extract relevant candidate's information mentioned in the following candidate's resume following the predefined properties.
                1) Please provide an accurate answers, no guessing.
                2) Please return 'N/A' string for all the information that is not mentioned. Do not return NaN.
                3) Extracted Properties of all Start date and End date:
                * if the month is not stated, assume that start/end date is in the middle of the year.
                * should never include english words such as 'months', 'years', 'days'. 
                * Instead, dates should be dates converted to the following format:
                * date values assigned are strictly in Python datetime format.
                Strict Format of either one: 
                    YYYY
                    YYYY-MM or YYYYMM
                    YYYY-MM-DD or YYYYMMDD
                4) Ensure that for any duration (year) calculation: 
                * Any end date that indicates "Present", refers to today's date, which is {current_date}. 
                * Do not assume the work experiences are continuous without breaks.
                * Method of duration calculation: Subtract the end date from start date to get the number of months. Finally sum up all relevant durations and convert to years. 
                * Triple check your calculations. ",
                """
            ),
            ("human", 
             """[Job Title] 
             {job_title}
             [Candidate's Resume]
             {text}   
             """),
        ]
    )

    #loader = PyPDFLoader(file_path, extract_images=True)
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
    runnable = prompt | llm.with_structured_output(schema=Candidate)
    result = runnable.invoke({"job_title":job_title,"text": documents,"current_date":datetime.now()})

    return result


def define_criteria(job_title,job_description,job_requirement,applicant_category):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert recruiting algorithm with 20 years experience in the recruiting industry. You will be provided with the job details (job title, applicant category, job description, job requirement). Execute all the tasks from step 1 to step 3 by strictly following the rules.\n"
                "\n[Tasks]\n"
                "1. Fill in relevant criteria's information based on the following job details with their properties.\n"
                "2. If the criteria are not specified, you should apply your hiring knowledge to suggest details to the criteria.\n"
                "3. Assign weightage to each of the criteria based on how important you feel they are in the job details.\n"
                "\n[Rules]\n"
                "- Make sure every criteria has one suggested detail.\n"
                "- Do not return 'Not Specified' as detail, suggest at least one detail based on common market hiring criteria.\n"
                "- You will penalized if you return 'Not Specified' as answer"
            ),
            ("human", 
            "[Job Details]\n"
            "Job Title : {job_title}\n"
            "Applicant Category : {applicant_category}\n"
            "[Start of Job Description] {job_description} [End of Job Description] \n "
            "[Start of Job Requirement] {job_requirement} [End of Job Requirement] "),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.4)
    runnable = prompt | llm.with_structured_output(schema=Job)
    result = runnable.invoke({"job_title":job_title,"job_description":job_description,"job_requirement":job_requirement,"applicant_category":applicant_category})

    criteria_data=[]
    weightage_data=[]
    
    for field_name in result.criteria[0].__fields__:
        criteria_data.append(getattr(result.criteria[0], field_name))
        
    for field_name in result.weightage[0].__fields__:
        weightage_data.append(getattr(result.weightage[0], field_name))

    df = pd.DataFrame({'details': criteria_data, 'weightage': weightage_data, 'selected':True})
    df.index = [x for x in result.criteria[0].__fields__]
    df.index.name='criteria'

    df.to_csv('criteria.csv')

    return df


def init_session(self, clear: bool =False):
    if not self.chat_inited or clear:
        st.session_state[self._session_key] = {}
        time.sleep(0.1)
        self.reset_history(self._chat_name)
