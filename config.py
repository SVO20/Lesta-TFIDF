import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ROOT_DIR, "db")

DB_NAME = "project.db"
DB_PATH = os.path.join("db", DB_NAME)
DB_URL = f"sqlite:///{DB_PATH}"
