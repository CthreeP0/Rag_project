import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from resume_screener import ResumeScreenerPack
from dotenv import load_dotenv
from utils import extract_information,get_table_download_link
from pdf_converter import doc2pdf
import secrets,string
import pandas as pd

load_dotenv(".env")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"FYP-Goo"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get('LANGCHAIN_API_KEY')

# Define your Streamlit app
def main():
    st.set_page_config(page_title="Chat with the Resume Parser", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Chat with the Resume Parser 💬🦙")
    st.info("Fill in this form before you start!", icon="📃")

    with st.sidebar:
        st.subheader('Document Chatbot!')

        cols = st.columns(1)
        if cols[0].button('Refresh'):
            print("session state before clearing",st.session_state)
            st.session_state.clear()
            print("session state after clearing",st.session_state)

    with st.form("my_form",clear_on_submit=False,border=True):
        st.session_state.batch_token = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
        st.title('Resume Parser Form')
        job_title = st.text_input(label=":rainbow[Job Title]", value="", on_change=None, placeholder="Insert your job title here", label_visibility="visible")
        job_description = st.text_area(label=":rainbow[Job Description]", value="", on_change=None, placeholder="Insert your job description here", label_visibility="visible")


        # Create a directory if it doesn't exist
        save_dir = f"uploaded_files/{st.session_state.batch_token}"
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
                                        "content": "These are the information that you've key in:"
                                                   f"Job Title: {job_title}"
                                                   f"Job Description: {job_description}"}]
        
        result_df = pd.DataFrame()
        

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                doc2pdf(save_dir)
                # Get a list of PDF files in the directory
                pdf_files = [filename for filename in os.listdir(save_dir) if filename.endswith(".pdf")]

                for filename in pdf_files:
                    result = extract_information(os.path.join(save_dir, filename))
                    candidate_dict = result.dict()
                    st.write(candidate_dict)
                    message = {"role": "assistant", "content": candidate_dict}
                    st.session_state.messages.append(message) # Add response to message history
                    # Convert the dictionary to a DataFrame
                    df = pd.DataFrame([candidate_dict])
                    # Append the result DataFrame to the main DataFrame
                    result_df = pd.concat([result_df, df], ignore_index=True)
                
                result_df.to_excel('results.xlsx')
                # Display the Excel file in the chat and provide a download link
                st.write("Resume Parsing is done for all resumes! You may download the results from the link below!")
                st.write(result_df)
                st.markdown(get_table_download_link(), unsafe_allow_html=True)
                # chat = ChatOpenAI(model='gpt-3.5-turbo-0125',temperature=0.4)
                # messages = [
                #     SystemMessage(
                #         content="""Please act as a hiring manager with 20 years experience. You will be provided a job title and its job description.\n
                #         [Instruction] 
                #         1. Determine all of the hiring criteria included in the job description.
                #         2. Assign weightage to each of the criteria based on its importance in the job position. The sum of the total weightage should be equals to 100
                #         3. Return them in a python dict.
                        
                #         [Format of the dict]
                #         [{Criterion : 'criterion',Weightage : weightage}]"""
                #     ),
                #     HumanMessage(
                #         content=f"""
                #         [Job Title]
                #         {job_title}

                #         [Job Description]
                #         {job_description}
                #                 """)]
                # response= chat.invoke(messages)


    # Initialize the chat messages history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Fill in this form before you start!"}]



    # @st.cache_resource(show_spinner=True)
    # def load_data(criteria,job_title,job_description):
    #     with st.spinner(text="Analyzing the documents..."):
    #         resume_path = "Ang Teik Hun Resume.pdf"
    #         job_description = job_description

    #         criteria_list = criteria
    #         result = []
    #         for criteria in criteria_list:
    #             resume_screener = ResumeScreenerPack(job_description=job_description, criteria=criteria)
    #             response = resume_screener.run(resume_path=resume_path)
    #             result.append(str(response))
    #         return result
    

    # if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    #     st.session_state.chat_engine = load_data


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