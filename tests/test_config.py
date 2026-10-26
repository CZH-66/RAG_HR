# tests/test_config.py
# -*- coding: utf-8 -*-
from config import BASE_DIR, CHROMA_DIR, COLLECTION_NAME, DATA_FILE, SIMILARITY_TOP_K


def test_project_layout():
    assert (BASE_DIR / "src").is_dir()
    assert (BASE_DIR / "data").is_dir()


def test_dataset_and_index_present():
    assert DATA_FILE.is_file()
    assert CHROMA_DIR.is_dir()
    assert COLLECTION_NAME
    assert SIMILARITY_TOP_K >= 1
