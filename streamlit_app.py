"""
Streamlit application.

Run as `streamlit run streamlit_app.py`
Running at `http://localhost:8501`

"""
import streamlit as st

from logger import info

info("Streamlit App code touched...")
st.title("Helloworld - The Streamlit App")
uploaded_file = st.file_uploader("Upload text file", type=["txt"])
if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")

    st.subheader("File content")
    st.text(file_contents)

    processed_text = file_contents.lower()  # example text -> to lowercase

    st.subheader("Processed text from file")
    st.text(processed_text)
