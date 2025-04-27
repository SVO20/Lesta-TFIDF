"""
SQLite-backed documents-corpus for the Streamlit TF-IDF demo.

- Three tables only: documents, lemmas, bridge table.
- SQLAlchemy Core, not ORM: clearer SQL.
- Compressed blobs – original text gzipped (lzma) to save space.
- Hash-index – xxhash64 keeps duplicates out faster.

"""

import math
import os
from typing import Dict, List

from sqlalchemy import (create_engine, MetaData, Table, Column, Integer, BigInteger,
                        LargeBinary, String, ForeignKey, Engine, Float, insert, select, delete, func)
from sqlalchemy.orm import sessionmaker

from config import DB_DIR, DB_PATH, DB_URL
from nlp import NlpDocContext


# --------------------------------------------------------------------------------------
# 0. Engine setup ----------------------------------------------------------------------
# --------------------------------------------------------------------------------------
def setup_database(use_existing: bool = True) -> Engine:
    """Create (or reuse) the SQLite file and hand back a SQLAlchemy *Engine*."""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    if not use_existing and os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # drop + recreate DB

    # Create the SQLAlchemy engine
    engine = create_engine(DB_URL, echo=False, future=True)
    return engine


# --------------------------------------------------------------------------------------
# 1. Schema (metadata is global singleton on import) -----------------------------------
# --------------------------------------------------------------------------------------

metadata = MetaData()

documents = Table("documents", metadata,
                  Column("doc_id", Integer, primary_key=True, autoincrement=True),
                  Column("xxhash64", BigInteger, nullable=False, index=True),
                  Column("compressed_text", LargeBinary, nullable=False))

lemmas = Table("lemmas", metadata,
               Column("lemma_id", Integer, primary_key=True, autoincrement=True),
               Column("lemma", String, unique=True, nullable=False))

documents_lemmas = Table("documents_lemmas", metadata,
                         Column("doc_id", Integer, ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True),
                         Column("lemma_id", Integer, ForeignKey("lemmas.lemma_id", ondelete="CASCADE"), primary_key=True),
                         Column("lemma_count", Integer, nullable=False),
                         Column("lemma_tf", Float, nullable=False))  # Float is equivalent to FLOAT UNSIGNED, both 4 bytes


# --------------------------------------------------------------------------------------
# 2. Public interface ------------------------------------------------------------------
# --------------------------------------------------------------------------------------
class Corpus:
    """Interface for the SQLite-powered text corpus using SQLAlchemy Core as DSL."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.Session = sessionmaker(bind=self.engine, future=True)
        metadata.create_all(self.engine)  # idempotent – safe on every import

    # ---------------------------------------------------------------- add / delete ---
    def add_document(self, nlp: NlpDocContext) -> int:
        """Puts a fully-parsed document into the three tables."""
        if not nlp.is_full():  # quick check
            raise ValueError("NlpDocContext must be filled before DB insertion.")

        with self.Session() as session:
            with session.begin():
                # 1) Insert document to `documents`
                res = session.execute(insert(documents)
                                      .values(xxhash64=nlp.xxhash64,
                                              compressed_text=nlp.compressed_text)
                                      .returning(documents.c.doc_id))
                doc_id = res.scalar_one()  # Get primary key of last added entry (first autoincremented doc_id)

                # 2) Upsert lemmas to `lemmas` (SQLite specific OR IGNORE)
                unique_lemmas = set(nlp.lemmas_count_map.keys())
                for lemma in unique_lemmas:
                    session.execute(insert(lemmas).prefix_with("OR IGNORE").values(lemma=lemma))

                # 3) Fetch lemma IDs
                rows = session.execute(select(lemmas.c.lemma_id, lemmas.c.lemma)
                                       .where(lemmas.c.lemma.in_(unique_lemmas)))  # each row is (lemma_id, lemma)
                lemmas_map = {row.lemma: row.lemma_id for row in rows}

                # 4) Fill associations table `documents_lemmas`
                doc_lemma_rows = []
                for curr_lemma in unique_lemmas:
                    doc_lemma_rows.append({"doc_id": doc_id,
                                           "lemma_id": lemmas_map[curr_lemma],
                                           "lemma_count": nlp.lemmas_count_map[curr_lemma],
                                           "lemma_tf": nlp.lemmas_tf_map[curr_lemma]})
                session.execute(insert(documents_lemmas), doc_lemma_rows)

        return doc_id

    def del_document(self, doc_id: int) -> None:
        """
        Delete a document from the database. Associated lemmas and associations will be deleted automatically
        due to the 'CASCADE' delete rule on the foreign key constraints.

        """
        with self.Session() as session:
            with session.begin():
                session.execute(delete(documents).where(documents.c.doc_id == doc_id))

    # ---------------------------------------------------------------- lookup -----------
    def get_hashmap(self) -> Dict[int, int]:
        """Returns `{xxhash64: doc_id, ...}` snapshot – used for duplicate detection."""
        with self.Session() as session:
            rows = session.execute(select(documents.c.doc_id, documents.c.xxhash64)).all()
            return {row.xxhash64: row.doc_id for row in rows} if rows else {}

    def lemmas_id_doccount_map(self):
        """For IDF: how many docs contain each lemma."""
        with (self.Session() as session):
            with session.begin():
                # Group by `lemma_id` -> arrgegate by count unique `doc_id`s
                query = select(
                    documents_lemmas.c.lemma_id,
                    # arrgegate by count `doc_id`s
                    func.count(func.distinct(documents_lemmas.c.doc_id)).label('doc_count')
                ).group_by(documents_lemmas.c.lemma_id)

                rows = session.execute(query).all()
                return {row.lemma_id: row.doc_count for row in rows}

    def lemmas_count(self, doc_id: int) -> Dict[int, int]:
        """Lemmas' counts for a given doc."""
        with self.Session() as session:
            with session.begin():
                query = select(
                    documents_lemmas.c.lemma_id,
                    documents_lemmas.c.lemma_count
                ).where(documents_lemmas.c.doc_id == doc_id)

                rows = session.execute(query).all()
                return {row.lemma_id: row.lemma_count for row in rows}

    def lemmas_tf(self, doc_id: int) -> Dict[int, float]:
        """Pre-computed term frequency map."""
        with (self.Session() as session):
            with session.begin():
                query = select(documents_lemmas.c.lemma_id, documents_lemmas.c.lemma_tf
                               ).where(documents_lemmas.c.doc_id == doc_id)

                rows = session.execute(query).all()
                return {row.lemma_id: row.lemma_tf for row in rows}

    # ---------------------------------------------------------------- TF-IDF -----------
    def lemma_tfidf_map(self, doc_id: int) -> Dict[int, float]:
        """TF-IDF per lemma for doc_id."""
        with self.Session() as session:
            with session.begin():
                query = select(func.count()).select_from(documents)
                total_docs = session.execute(query).scalar_one()
                if total_docs == 0:  # empty corpus – nothing to do
                    return {}

                tf_map = self.lemmas_tf(doc_id)
                if not tf_map:
                    raise RuntimeError(f"document {doc_id=} has no lemmas (should never happen ??!)")

                lemid_doccount_map = self.lemmas_id_doccount_map()

                tf_idf_result: Dict[int, float] = {}
                for lemma_id, tf in tf_map.items():
                    doccount = lemid_doccount_map.get(lemma_id) or 0
                    if doccount == 0:
                        continue  # ownerless lemma encoutered - infinite IDF case skipped

                    idf = math.log(total_docs / doccount)
                    tf_idf_result[lemma_id] = tf * idf

                return tf_idf_result

    # ---------------------------------------------------------------- output ----------
    def document_lemmas_info(self, doc_id: int) -> List[dict]:
        """Compose a list of lemma stats, ready for convert to DataFrame.

        lemma_stat = {'lemma': str, 'count': int, 'tf': float, 'idf': float, 'tf-idf': float}
        Sorted descending by 'tf-idf'.
        """
        with self.Session() as session:
            with session.begin():
                # Total documents
                total_docs = session.execute(select(func.count()).select_from(documents)).scalar_one()
                if total_docs == 0:
                    return []

                # Load known info
                tf_map = self.lemmas_tf(doc_id)
                count_map = self.lemmas_count(doc_id)
                tfidf_map = self.lemma_tfidf_map(doc_id)
                lemid_doccount_map = self.lemmas_id_doccount_map()

                # Resolve lemma_id -> lemma
                lemma_ids = list(count_map.keys())
                rows = session.execute(select(lemmas.c.lemma_id, lemmas.c.lemma)
                                       .where(lemmas.c.lemma_id.in_(lemma_ids))).fetchall()
                id_to_lemma = {row.lemma_id: row.lemma for row in rows}

                # Compose list of stats
                data: List[dict] = []
                for lemma_id in lemma_ids:
                    # Assemble one dict per lemma
                    lemma = id_to_lemma.get(lemma_id, f"[id={lemma_id}]")
                    count = count_map.get(lemma_id, 0)
                    tf = tf_map.get(lemma_id, 0.0)
                    docs_count = lemid_doccount_map.get(lemma_id, 1)
                    idf = math.log(total_docs / docs_count) if docs_count else 0.0  # idf of lemma iside Corpus
                    tfidf = tfidf_map.get(lemma_id, 0.0)

                    data.append({"word": lemma,
                                 "count": count,
                                 "tf": tf,
                                 "idf": idf,
                                 "tf-idf": tfidf})

                # Pre-sort by tf-idf descending
                return sorted(data, key=lambda x: x["tf-idf"], reverse=True)
