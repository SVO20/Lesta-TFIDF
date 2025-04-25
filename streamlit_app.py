"""
Streamlit application.

Run as `streamlit run streamlit_app.py`
Running at `http://localhost:8501`

"""
import streamlit as st

import tfidf
from logger import info
from tfidf import compute_tf

info("Streamlit App code touched...")
st.title("Lesta TD-IFD Analyser - The Streamlit App")
uploaded_file = st.file_uploader("Upload the text file to analyze below: ", type=["txt"])
if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")

    # st.subheader("File content")
    # st.text(file_contents)
    # processed_text = " ".join(tfidf.tokenize(file_contents))
    tokens = tfidf.tokenize(file_contents)

    df_tf = compute_tf(tokens, 50)

    st.subheader("Terms Frequency for the given text (Russian only)")
    df_tf.index = df_tf.index + 1 # make index 1-based for display
    st.dataframe(df_tf)

    # st.subheader("Bare russian tokens got from text file")
    # st.text(processed_text)
