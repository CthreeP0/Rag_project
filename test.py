import streamlit as st
import pandas as pd


fn = 'criteria.csv'
df = pd.read_csv(fn)
update = st.data_editor(df, num_rows='dynamic', key='df')

if st.button('Save dataframe'):
    update.to_csv(fn, index=False)

st.write(st.session_state.df)
