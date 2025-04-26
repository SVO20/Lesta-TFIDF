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
    st.title("TD-IFD Analyser")
    choice = st.radio("База данных:",
                      ("Использовать существующую базу", "Создать новую базу"),
                      index=0)
    if st.button("Подтвердить выбор базы данных"):
        st.session_state.db_ready = True
        st.session_state.use_existing_db = (choice == "Использовать существующую базу")
        # Initialize empty hash map
        st.session_state.hashmap = {}
        st.rerun()
    else:
        st.warning('Надо выбрать пункт')
        st.stop()

# ========================== Setup (run once) =========================================
if 'engine' not in st.session_state:
    # Setup database first time
    st.session_state.engine = setup_database(use_existing=st.session_state.use_existing_db)
engine = st.session_state.engine
if 'corpus' not in st.session_state:
    # Setup Corpus first time
    st.session_state.corpus = Corpus(st.session_state.engine)
corpus = st.session_state.corpus
if 'hashmap' not in st.session_state:
    # Get hashmap first time
    st.session_state.hashmap = corpus.get_hashmap()
hashmap = st.session_state.hashmap

# Needed for correct change streamlit uploader widgets one-by-one for new file(s)
if 'uploader_round' not in st.session_state:
    # integer to give each uploader a unique key
    st.session_state.uploader_round = 0

if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = []    # list of {"filename": str, "doc_id": int, "Hash": int}
    st.session_state.last_uploaded = None        # name of the last file


# ======== File Storing =========
def store_file(uploaded_file):
    file_contents = uploaded_file.read().decode("utf-8")
    nlp = NlpDocContext(file_contents)
    try:
        # Compute document hash
        hash_ = hash_original_text(nlp)
        # Check duplicate in global hash map
        if nlp.xxhash64 in hashmap:
            doc_id = hashmap[nlp.xxhash64]
            info(f"Duplicate document detected, using existing doc_id={doc_id}")
            return doc_id, hash_

        # New document: add to database and update hash map
        tokenize(nlp)
        compute_tf(nlp)
        compress_original_text(nlp)
        doc_id = corpus.add_document(nlp)
        # Add to the hashmap explicitly from here
        hashmap[nlp.xxhash64] = doc_id
        info(f"Document added with doc_id={doc_id}, hash={nlp.xxhash64}")
    except Exception as e:
        st.error(f"Ошибка при работе с базой данных: {e}")
        raise e

    finally:
        nlp.clear()

    if doc_id is None:
        raise RuntimeError("doc_id recieved from database invalid!")
    else:
        return doc_id, hash_


# === This round ===

# Create a unique key for this run round of the uploader
# to say Streamlit display fresh upload widget
uploader_key = f"file_uploader_{st.session_state.uploader_round}"

# ========================== File upload section =========================================
# Compose form with the submit button
with st.form(key="upload_form"):
    st.title("Добавление документа для анализа")
    files = st.file_uploader(label="Загрузите один или несколько текстовых файлов (.txt)",
                             type=["txt"],
                             accept_multiple_files=True,
                             key=uploader_key) # <- every round new one
    submit = st.form_submit_button("Submit")

# Pass on
if submit and files:
    last_filename = ""
    for f in files:


        doc_id, hash_ = store_file(f)

        # Add to the existing list (do not overwrite)
        st.session_state.uploaded_files_info.append({"filename": f.name, "doc_id": doc_id, "hash": hash_})
        last_filename = f.name

    st.session_state.last_uploaded = last_filename

    # Prepare for a bnew uploader next time
    st.session_state.uploader_round += 1

    # Force Streamlit to rerun, so the uploader clears itself
    st.rerun()

# ================= Display section ======================================================
if st.session_state.uploaded_files_info:
    st.info(f"Последний загруженный файл был: **{st.session_state.last_uploaded}**")
    df = pd.DataFrame(st.session_state.uploaded_files_info)
    df.index += 1                       # make index 1-based
    df.columns = ["Filename", "doc_id", "Hash"]  # nicer column names
    st.table(df)


# -----------------------------------------------------------------------------------------------
def some(uploaded_file):
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