"""
Streamlit application.

Run as `streamlit run streamlit_app.py`
Running at `http://localhost:8501`

"""
import humanize
import pandas as pd
import streamlit as st

from database import setup_database, Corpus
from logger import info
from nlp import NlpDocContext, hash_original_text, tokenize, compute_tf, compress_original_text


if 'db_ready' not in st.session_state:
    st.title("Lesta TD-IFD Analyser")
    choice = st.radio(
        "База данных:",
        ("Использовать существующую базу", "Создать новую базу"),
        index=0
    )
    if st.button("Подтвердить выбор базы данных"):
        st.session_state.db_ready = True
        st.session_state.use_existing_db = (choice == "Использовать существующую базу")
        # Initialize empty hash map
        st.session_state.hashmap = {}
        st.rerun()
    else:
        st.stop()

info("Streamlit App code touched...")
st.title("Lesta TD-IFD Analyser - The Streamlit App")

# One setup database
if 'engine' not in st.session_state:
    st.session_state.engine = setup_database(use_existing=st.session_state.use_existing_db)
if 'corpus' not in st.session_state:
    st.session_state.corpus = Corpus(st.session_state.engine)

engine = st.session_state.engine
corpus = st.session_state.corpus

# One setup hashmap
if 'hashmap' not in st.session_state:
    st.session_state.hashmap = corpus.get_hashmap()

hashmap = st.session_state.hashmap

# ========================== File upload section =========================================
st.title("Добавление документа для анализа")
uploaded_file = st.file_uploader("Загрузите текстовый файл (.txt)", type=["txt"])

if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")
    nlp = NlpDocContext(file_contents)
    try:
        # Compute document hash
        hash_original_text(nlp)
        # Check duplicate in global hash map
        if nlp.xxhash64 in hashmap:
            doc_id = hashmap[nlp.xxhash64]
            info(f"Duplicate document detected, using existing doc_id={doc_id}")
            info("Drop test!")
        else:
            # New document: add to database and update hash map
            tokenize(nlp)
            compute_tf(nlp)
            compress_original_text(nlp)
            doc_id = corpus.add_document(nlp)
            # Add to the hashmap
            hashmap[nlp.xxhash64] = doc_id
            info(f"Document added with doc_id={doc_id}, hash={nlp.xxhash64}")

        # Convert the count dict to DataFrame
        df = pd.DataFrame(nlp.lemmas_tf_map.items(), columns=['word', 'tf'])
        df_sorted = df.sort_values(by="tf", ascending=False).head(50).reset_index(drop=True)

        df_sorted.index = df_sorted.index + 1  # make index 1-based for display
        st.subheader("Terms Frequency for the given text (Russian only)")
        st.dataframe(df_sorted)

        st.subheader("Документ загружен")
        st.code(f"Hash: {nlp.xxhash64}", language="text")
        st.text_area("Первые 100 символов текста:", value=file_contents[:100], height=150)

        st.subheader("Info:")
        st.info(f"Размер оригинального файла: {humanize.naturalsize(uploaded_file.size)}\n\n"
                f"Хранение текста в базе данных будет стоить: {humanize.naturalsize(len(nlp.compressed_text))}")

    except Exception as e:
        st.error(f"Ошибка при работе с NLP/БД: {e}")
    finally:
        nlp.clear()
        # clear
        st.session_state["uploader"] = None
        st.rerun()
