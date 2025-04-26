"""
Streamlit application.

Run as `streamlit run streamlit_app.py`
Running at `http://localhost:8501`

"""
import pandas as pd
import streamlit as st

from database import setup_database, Corpus
from logger import info
from nlp import NlpDocContext, hash_original_text, tokenize, compute_count_tf, compress_original_text

if 'db_ready' not in st.session_state:
    st.title("TD-IFD Analyser")
    choice = st.radio("База данных:", ("Использовать существующую базу",
                                       "Создать новую базу"),
                      index=1)
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
    st.session_state.uploaded_files_info = []  # list of {"filename": str, "doc_id": int, "Hash": int}
    st.session_state.last_uploaded = None  # name of the last file

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
                             key=uploader_key)  # <- every round new one
    submit = st.form_submit_button("Submit")

# Pass on

if submit and files:
    # Load files
    documents_loaded = []
    last_filename = ""
    last_doc_id = None
    for f in files:
        file_contents = f.read().decode("utf-8")
        if not file_contents:
            continue

        # --- Document prepare section ---

        doc_cxt = NlpDocContext(file_contents)
        try:

            xxhash64 = hash_original_text(doc_cxt)
            if xxhash64 in hashmap:
                doc_id = hashmap[xxhash64]
                st.info(f"File {f.name} has same content as existing document doc_id={doc_id}.")
                documents_loaded.append(doc_id)
                last_filename = f.name
                last_doc_id = doc_id
                continue

            tokens = tokenize(doc_cxt)
            if not tokens:
                st.warning(f"File {f.name} has no valid tokens. To drop.")
                continue

            lemmas_count_map, lemmas_tf_map = compute_count_tf(doc_cxt)
            if lemmas_count_map is None or lemmas_tf_map is None:
                st.error(f"File {f.name} gives incorrect results during analysis. To drop.")
                continue

            compress_original_text(doc_cxt)

            # Ready for database -> to database
            doc_id = corpus.add_document(doc_cxt)

            # Update the hashmap explicitly from here
            hashmap[xxhash64] = doc_id
            info(f"File {f.name} prepared and added to database with doc_id={doc_id}")

        except Exception:
            raise
        finally:
            doc_cxt.clear()

        last_filename = f.name
        last_doc_id = doc_id

        # --- Document prepare section END ---

    if last_doc_id is None:
        st.error('last_doc_id is None !!?')
        st.stop()
    else:
        st.session_state.last_uploaded = (last_filename, last_doc_id)

    # Prepare for a new uploader next time
    st.session_state.uploader_round += 1
    # Force Streamlit to rerun, so the uploader clears itself
    st.rerun()

# ================= Display section ======================================================
if st.session_state.last_uploaded:
    st.info(f"Последний выбранный файл был: **{st.session_state.last_uploaded[0]}**\n\n"
            f"Ниже представлен анализ его содержания.\n\n\n\n"
            f"Для анализа другого файла выберите его в диалоге выше, он будет либо загружен, либо взят из базы данных.")

    last_filename, last_doc_id = st.session_state.last_uploaded

    lemmas_info = corpus.document_lemmas_info(last_doc_id)
    df = pd.DataFrame(lemmas_info)

    # Custom sorting
    sort_field = st.selectbox("Выберите поле для сортировки:",
                              options=["tf-idf", "count", "tf", "idf"],
                              index=0)  # default: tf-idf
    ascending = st.checkbox("Сортировать по возрастанию значения?", value=False)
    df = df.sort_values(by=sort_field, ascending=ascending)

    df.index = df.index + 1  # make index 1-based for display
    df = df.head(50)

    # Display as table
    st.table(df)
