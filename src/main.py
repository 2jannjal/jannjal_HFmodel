##First Commit
import streamlit as st
from transformers_utils import return_score

st.title("Similarity Score by Sentence")
st.write("Insert two senteces below and generate a score based on a model")

container = st.container(border=True)
with container:
    col1, col2 = st.columns(2)

    #Comparison Sentence 1
    with col1:
        senone = st.text_input("Sentence 1")

    #Comparison Sentence 2
    with col2:
        sentwo = st.text_input("Sentence 2")

    col3, col4, col5 = st.columns(3)

    ### Selectbox Options
    model_options = [
        "bert-base-uncased",
        "all-MiniLM-L6-v2",
        "roberta-large"
    ]

    with col3:
        selected_model_from_select = st.selectbox("Please Select a Model", options= model_options)



sub_button = st.button("Submit")

if sub_button:
    selected_model = selected_model_from_select

    score = return_score(senone, sentwo, selected_model)

    with col5:
        # Convert the similarity score to a percentage
        score_percentage = score * 100

        # Display the score in a larger font size using HTML in Markdown
        st.markdown(f'<h2 style="font-size: 24px;">Score: {score_percentage:.2f}%</h2>', unsafe_allow_html=True)
