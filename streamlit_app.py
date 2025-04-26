"""
Streamlit application.

Run as `streamlit run streamlit_app.py`
Running at `http://localhost:8501`

"""
import humanize
import pandas as pd
import streamlit as st

from logger import info
from nlp import NlpDocContext, hash_original_text, tokenize, compute_tf, compress_original_text

CORPUS_HASHMAP = {}  # for the quck duplicates detection   {xxh64: doc_id}

info("Streamlit App code touched...")
st.title("Lesta TD-IFD Analyser - The Streamlit App")
uploaded_file = st.file_uploader("Upload the text file to analyze below: ", type=["txt"])
if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")

    nlp = NlpDocContext(file_contents)
    try:
        CORPUS_HASHMAP.update({hash_original_text(nlp): 'doc1'})
        assert tokenize(nlp)
        dict_tf = compute_tf(nlp)

        # Convert the count dict to DataFrame
        df = pd.DataFrame(dict_tf.items(), columns=['word', 'tf'])
        df_sorted = df.sort_values(by="tf", ascending=False).head(50).reset_index(drop=True)

        df_sorted.index = df_sorted.index + 1  # make index 1-based for display
        st.subheader("Terms Frequency for the given text (Russian only)")
        st.dataframe(df_sorted)

        compressed_size = len(compress_original_text(nlp))
        st.subheader("Info:")
        st.info(f"Размер оригинального файла: {humanize.naturalsize(uploaded_file.size)}\n\n"
                f"Хранение текста в базе данных будет стоить: {humanize.naturalsize(compressed_size)}")
    except:
        raise RuntimeError(f"Error during NLP-workflow on file '{uploaded_file.name}'")
    finally:
        nlp.clear()
