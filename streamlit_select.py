import streamlit as st
import secrets
import string
import pandas as pd
from utils import define_criteria

st.set_page_config(page_title="Chat with the Resume Parser", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the Resume Parser 💬🦙")
st.info("Fill in this form before you start!", icon="📃")
df = pd.DataFrame()

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
    job_requirement = st.text_area(label=":rainbow[Job Requirement]", value="", on_change=None, placeholder="Insert your job requirement here", label_visibility="visible")
    applicant_category = st.selectbox(
        'Which applicant category does this job belong to?',
        ('Entry-Level', 'Mid-Level', 'Managerial-Level'))

    st.write('You selected:', applicant_category)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

if submitted:
    st.session_state.messages = [{"role": "assistant", 
                                    "content": f"""These are the information that you've key in:\nJob Title: {job_title}\nJob Description:\n{job_description}\nJob Requirement:\n{job_requirement}\nApplicant Category: {applicant_category}""",
                                    "type":"message"}]
    
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = define_criteria(job_title,job_description,job_requirement,applicant_category)
            # Extract criteria and weightage
            criteria_data = []
            weightage_data = []

            for field_name in result.criteria[0].__fields__:
                criteria_data.append(getattr(result.criteria[0], field_name))
                
            for field_name in result.weightage[0].__fields__:
                weightage_data.append(getattr(result.weightage[0], field_name))

            # Create DataFrame
            df = pd.DataFrame({'criteria': criteria_data, 'weightage': weightage_data})

            # # Set the name of the criteria as the index
            df.index = [x for x in result.criteria[0].__fields__]

            message = {"role": "assistant", "content": result_df,"type":'dataframe'}
            st.session_state.messages.append(message)


# options = st.multiselect(
#     'Which criteria would you like to match/assess?',
#     ['Education Background', 'CGPA', 'Skill Groups', 'Year of Total Working Experience',''],
#     ['Education Background', 'CGPA', 'Skill Groups'])

# st.write('You selected:', options)

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Fill in this form before you start!","type":"message"}]

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt,"type":"prompt"})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        if message["type"] == 'pre_ranking_extracted_result':
            with open("results.xlsx","rb") as extracted_results:
                st.download_button(label="Download as Excel",data=extracted_results,file_name=f"{st.session_state.batch_token}.xlsx")
        else:
            st.write(message["content"])
