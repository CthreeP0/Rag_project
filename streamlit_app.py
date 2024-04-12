import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from resume_screener import ResumeScreenerPack
from dotenv import load_dotenv
from utils import extract_information,define_criteria
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

    with st.form("my_form",clear_on_submit=False,border=True):
        st.session_state.batch_token = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
        st.title('Resume Parser Form')
        job_title = st.text_input(label=":rainbow[Job Title]", value="", on_change=None, placeholder="Insert your job title here", label_visibility="visible")
        job_description = st.text_area(label=":rainbow[Job Description]", value="", on_change=None, placeholder="Insert your job description here", label_visibility="visible")
        job_requirement = st.text_area(label=":rainbow[Job Requirement]", value="", on_change=None, placeholder="Insert your job requirement here", label_visibility="visible")
        applicant_category = st.selectbox(
            'Which applicant category does this job belong to?',
            ('Entry-Level', 'Mid-Level', 'Managerial-Level'))

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
        st.session_state["button_pressed"] = True
        result_df = pd.DataFrame()
        df = define_criteria(job_title,job_description,job_requirement,applicant_category)
        

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                doc2pdf(save_dir)
                # Get a list of PDF files in the directory
                pdf_files = [filename for filename in os.listdir(save_dir) if filename.endswith(".pdf")]

                for filename in pdf_files:
                    result = extract_information(os.path.join(save_dir, filename),job_title)
                    candidate_dict = result.dict()
                    st.write(candidate_dict)
                    # Convert the dictionary to a DataFrame
                    df = pd.DataFrame([candidate_dict])
                    # Append the result DataFrame to the main DataFrame
                    result_df = pd.concat([result_df, df], ignore_index=True)
                

                # Display the Excel file in the chat and provide a download link
                result_df.to_excel('results.xlsx')
                st.session_state.messages = [{"role": "assistant", "content": f"Resume Parsing is done for all resumes! You may download the results from the link below!","type":'message'}]
                message = {"role": "assistant", "content": result_df,"type":'dataframe'}
                st.session_state.messages.append(message)

    if "button_pressed" in st.session_state.keys():
        with st.sidebar:
            st.subheader('Define your evaluation criteria here!')

            cols = st.columns(1)
            if cols[0].button('Refresh'):
                print("session state before clearing",st.session_state)
                st.session_state.clear()
                print("session state after clearing",st.session_state)

            # Create DataFrame
            df = pd.read_csv('criteria.csv')
            edited_df = st.data_editor(df,num_rows='dynamic', key='df')

            favorite_command = edited_df.loc[edited_df["weightage"].idxmax()]["criteria"]
            st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

            if st.button('Save dataframe'):
                edited_df.to_csv('criteria.csv', index=False)

            st.write(st.session_state.df)

    # Initialize the chat messages history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Fill in this form before you start!","type":"message"}]


    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt,"type":"prompt"})

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
                message = {"role": "assistant", "content": response,"type":"message"}
                st.session_state.messages.append(message) # Add response to message history

if __name__ == "__main__":
    main()
