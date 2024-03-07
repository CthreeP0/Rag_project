import streamlit as st
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from utils import extract_information
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

# backwards compatibility
try:
    from llama_index.core.llms.llm import LLM
except ImportError:
    from llama_index.core.llms.base import LLM


openai.api_key = os.getenv("OPENAI_API_KEY")

# Define your data models
class CriteriaDecision(BaseModel):
    """The decision made based on a single criteria."""

    scoring: int = Field(description="The scoring in the scale of 1-10 made based on the criteria")
    reasoning: str = Field(description="The reasoning behind the decision")

class ResumeScreenerDecision(BaseModel):
    """The decision made by the resume screener."""
    
    criteria_decisions: List[CriteriaDecision] = Field(description="The decisions made based on the criteria")
    overall_reasoning: str = Field(description="The reasoning behind the overall decision")
    scoring: int = Field(description="The overall scoring in the scale of 1-10 made based on the overall criteria")


QUERY_TEMPLATE = """
[Instruction] You will be provided with details such as the screening criteria, job description and job requirements.
Please act as an impartial judge and evaluate the candidate's {criteria_str} based on the job description and job requirements. For this evaluation, you should primarily consider the following accuracy:
[Accuracy]
Score 1: The candidate's {criteria_str} is completely unrelated to job_description .
Score 3: The candidate's {criteria_str} has minor relevance but does not align with job_description.
Score 5: The candidate's {criteria_str} has moderate relevance but contains inaccuracies to job_description.
Score 7: The candidate's {criteria_str} aligns with job_description. but has minor errors or omissions on either one of them.
Score 10: The candidate's {criteria_str} is completely accurate and aligns very well with job_description.
        

### Job Description
{job_description}

### Screening Criteria
{criteria_str}
"""

# Define your ResumeScreenerPack class
class ResumeScreenerPack(BaseLlamaPack):
    def __init__(self, job_description: str, criteria: str, llm: Optional[LLM] = None) -> None:
        self.reader = extract_information
        llm = llm or OpenAI(model="gpt-3.5-turbo-0125")
        service_context = ServiceContext.from_defaults(llm=llm)
        self.synthesizer = TreeSummarize(output_cls=ResumeScreenerDecision, service_context=service_context)
        self.query = QUERY_TEMPLATE.format(job_description=job_description, criteria_str=criteria)

    def get_modules(self) -> Dict[str, Any]:
        return {"reader": self.reader, "synthesizer": self.synthesizer}

    def run(self, resume_path: str, *args: Any, **kwargs: Any) -> Any:
        docs = self.reader(resume_path)
        output = self.synthesizer.synthesize(query=self.query, nodes=[NodeWithScore(node=doc, score=1.0) for doc in docs])
        return output.response

# Define your Streamlit app
def main():
    st.set_page_config(page_title="Chat with the Resume Parser Chatbot", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Chat with the Resume Parser Chatbot 💬🦙")
    st.info("Fill in this form before you start!", icon="📃")
    with st.form("my_form",clear_on_submit=False,border=True):
        st.title('Resume Parser Form')
        job_title = st.text_input(label=":rainbow[Job Title]", value="", on_change=None, placeholder="Insert your job title here", label_visibility="visible")
        job_description = st.text_area(label=":rainbow[Job Description]", value="", on_change=None, placeholder="Insert your job description here", label_visibility="visible")


        # Create a directory if it doesn't exist
        save_dir = "uploaded_files"
        os.makedirs(save_dir, exist_ok=True)

        # File uploader
        uploaded_files = st.file_uploader("Drop your resume here", accept_multiple_files=True)

        # Check if files are uploaded
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                file_name = uploaded_file.name
                file_path = os.path.join(save_dir, file_name)
                
                # Write the contents of the file to a new file
                with open(file_path, "wb") as f:
                    f.write(bytes_data)
                
                # Provide feedback to the user
                st.write("File saved:", file_name)
        else:
            print("FAILED")
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
    if submitted:
        st.session_state.messages = [{"role": "assistant", 
                                        "content": f'''These are the information that you've key in:\n
                                                    Job Title: {job_title},Job Description: {job_description}\n
                                                    '''}]
        

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat = ChatOpenAI(model='gpt-3.5-turbo-0125',temperature=0.4)
                messages = [
                    SystemMessage(
                        content="""Please act as a hiring manager with 20 years experience. You will be provided a job title and its job description.\n
                        [Instruction] 
                        1. Determine all of the hiring criteria included in the job description.
                        2. Assign weightage to each of the criteria based on its importance in the job position. The sum of the total weightage should be equals to 100
                        3. Return them in a python dict.
                        
                        [Format of the dict]
                        [{Criterion : 'criterion',Weightage : weightage}]"""
                    ),
                    HumanMessage(
                        content=f"""
                        [Job Title]
                        {job_title}

                        [Job Description]
                        {job_description}
                                """)]
                response= chat.invoke(messages)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message) # Add response to message history

    # Initialize the chat messages history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Fill in this form before you start!"}]



    @st.cache_resource(show_spinner=True)
    def load_data(criteria,job_title,job_description):
        with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
            resume_path = "Ang Teik Hun Resume.pdf"
            job_description = job_description

            criteria_list = criteria
            result = []
            for criteria in criteria_list:
                resume_screener = ResumeScreenerPack(job_description=job_description, criteria=criteria)
                response = resume_screener.run(resume_path=resume_path)
                result.append(str(response))
            return result
    

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = load_data


    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def string_to_list(input_string):
        return input_string.split(", ")

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                criteria_list = string_to_list(prompt)
                response = st.session_state.chat_engine(criteria_list,job_title,job_description)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message) # Add response to message history

if __name__ == "__main__":
    main()
