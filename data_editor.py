import streamlit as st
import pandas as pd
import string,secrets
from utils import define_criteria

st.set_page_config(page_title="Chat with the Resume Parser", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the Resume Parser 💬🦙")
st.info("Fill in this form before you start!", icon="📃")

with st.form("my_form",clear_on_submit=False,border=True):
    st.session_state.batch_token = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
    st.title('Resume Parser Form')
    st.session_state.job_title = st.text_input(label=":rainbow[Job Title]", value="", on_change=None, placeholder="Insert your job title here", label_visibility="visible")
    st.session_state.job_description = st.text_area(label=":rainbow[Job Description]", value="", on_change=None, placeholder="Insert your job description here", label_visibility="visible")
    st.session_state.job_requirement = st.text_area(label=":rainbow[Job Requirement]", value="", on_change=None, placeholder="Insert your job requirement here", label_visibility="visible")
    st.session_state.applicant_category = st.selectbox(
        'Which applicant category does this job belong to?',
        ('Entry-Level', 'Mid-Level', 'Managerial-Level'))

    st.write('You selected:', st.session_state.applicant_category)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state["button_pressed"] = True
        df = define_criteria(st.session_state.job_title,st.session_state.job_description,st.session_state.job_requirement,st.session_state.applicant_category)

 
    st.session_state.messages = [{"role": "assistant", 
                                    "content": f"""These are the information that you've key in:
                                    Job Title: {st.session_state.job_title}
                                    Job Description: {st.session_state.job_description}
                                    Job Requirement: {st.session_state.job_requirement}
                                    Applicant Category: {st.session_state.applicant_category}""",
                                    "type":"message"}]


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

        # # Every form must have a submit button.
        # criteria_submitted = st.form_submit_button("Save")

        # if criteria_submitted:
        

        st.write(st.session_state.df)

        

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
        elif message["type"] == 'dataframe':
            st.write(message["content"])
        else:
            st.write(message["content"])

# if "edited_df" in st.session_state.keys():
#     print(st.session_state.edited_df)
#     print(st.session_state.df)
#     st.session_state.messages.append({"role": "assistant", "content": st.session_state.edited_df,"type":"edited_df"})