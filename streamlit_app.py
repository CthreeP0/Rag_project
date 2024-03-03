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

# backwards compatibility
try:
    from llama_index.core.llms.llm import LLM
except ImportError:
    from llama_index.core.llms.base import LLM




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
Score 1: The candidate's {criteria_str} is completely unrelated to job_description and job_requirements.
Score 3: The candidate's {criteria_str} has minor relevance but does not align with job_description and job_requirements.
Score 5: The candidate's {criteria_str} has moderate relevance but contains inaccuracies to job_description and job_requirements.
Score 7: The candidate's {criteria_str} aligns with job_description and job_requirements. but has minor errors or omissions on either one of them.
Score 10: The candidate's {criteria_str} is completely accurate and aligns very well with job_description and job_requirements.
        

### Job Description
{job_description}

### Job Requirements
{job_requirements}

### Screening Criteria
{criteria_str}
"""

# Define your ResumeScreenerPack class
class ResumeScreenerPack(BaseLlamaPack):
    def __init__(self, job_description: str, job_requirements: str, criteria: str, llm: Optional[LLM] = None) -> None:
        self.reader = extract_information
        llm = llm or OpenAI(model="gpt-3.5-turbo-0125")
        service_context = ServiceContext.from_defaults(llm=llm)
        self.synthesizer = TreeSummarize(output_cls=ResumeScreenerDecision, service_context=service_context)
        self.query = QUERY_TEMPLATE.format(job_description=job_description, job_requirements=job_requirements, criteria_str=criteria)

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
    st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

    # Initialize the chat messages history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Ask any question you want!"}]

    @st.cache_resource(show_spinner=True)
    def load_data(criteria):
        with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
            resume_path = "resume/JiaHao_Lo.pdf"
            job_description = """Responsible for design, planning, and coordinating the implementation of Data Science work activities in the Group Digital with established structured processes and procedures to support PETRONAS's digital agenda.
             

            1) Technical & Professional Excellence

            Responsible for ensuring data required for analytic models is of required quality and models are constructed to standards and deployed effectively.
            Implement data science industry best practices, relevant cutting-edge technology, and innovation in projects to ensure the right solutions fulfill the business requirements.

            2) Technical/Skill Leadership & Solutioning

            Responsible for developing appropriate technical solutions to address business pain points with insights and recommendations.
            Implement an established analytics strategy by adopting the right technologies and technical requirements when executing projects to ensure business value generation.
            Execute operational excellence through continuous technical and process improvement initiatives within projects to improve operations efficiency and effectiveness within own activity & projects.

            3) Technical Expertise

            Track and follow up with relevant parties to ensure Technical Excellence Programmes are successfully deployed and integrated into work processes, documents, policies, and guidelines.
            Participate in a community of practices and network with internal and external technical experts by identifying solutions to common problems and capturing and sharing existing data science knowledge for continuous excellence.

             

            Be part of our DS team in at least one of the following areas:

            Machine Learning

            Roles: Design analytics solutions for business problems; develop, evaluate, optimize, deploy and maintain models.

            Tech stack: ML Algorithms, Python, SQL, Spark, Git, Cloud Services, Deep Learning frameworks, MLOps, etc

             

            Natural Language Processing

            Roles: Design text analytics solutions for business problems; develop, evaluate, optimize, deploy and maintain text processing and analytics solutions.

            Tech stack: Python, SQL, Git, NLTK, Deep Learning frameworks, MLOps, Text analytics, NLP, NLU, NLG, Language Models, etc

             

            Computer Vision

            Roles: Design Image and video analytics solutions for business problems; develop, evaluate, optimize, deploy and maintain solutions

            Tech stack: Tensorflow, OpenCV, Fastai, Pytorch, MLFlow, Spark, MLlib Python, SQL, Git, Deep Learning frameworks, MLOps, etc

             

            Optimization / Simulation

            Roles: Design optimization/simulation analytics solutions for business problems; develop, evaluate, optimize, deploy and maintain solutions

            Tech stack: mathematical/process models, Simulation modeling, AnyLogic, Simio, mixed-integer programming (linear and nonlinear), Python, Pyomo, Gurobi solver, MLOps, etc."""

            job_requirements = """
            BS or MS in Data Science 5 to 15 years of experience in electronic package design/ mechanical enclosure design for hand held products.
            """
            criteria_list = criteria
            result = []
            for criteria in criteria_list:
                resume_screener = ResumeScreenerPack(job_description=job_description, job_requirements=job_requirements, criteria=criteria)
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
                response = st.session_state.chat_engine(criteria_list)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message) # Add response to message history

if __name__ == "__main__":
    main()