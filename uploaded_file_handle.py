from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import fitz
from embedding import embed_chunk_strings, update_cache_dict
import numpy as np


#   WILL update session state uploaded_files_embeddings with any new uploaded files that are passed to this function
def convert_docs_to_dict_chunks(uploaded_files: list[UploadedFile]) -> dict[str, dict[str, np.ndarray]]:
    chunk_dict: dict[str, dict[str, np.ndarray]] = {}
    try:
        if not uploaded_files:
            raise Exception("No uploaded files detected in convert_doc_to_dict_chunks in document_handle.py")
        for file in uploaded_files:

            file_name = file.name
            # st.info(f"Current Session State of uploaded_files_embeddings BEFORE UPLOADING: {st.session_state["uploaded_files_embeddings"]}")
            st.info(file_name)
            if st.session_state["uploaded_files_embeddings"]: 
                for f_name in st.session_state["uploaded_files_embeddings"].keys():
                    if (f_name == file_name):
                        st.info("Name alreading in embeddings! Will not recalculate Embeddings")
                        return chunk_dict
            file_string = extract_text_from_file(file)
            file_chunks = create_chunks_from_string(file_string)
            chunk_embedding_dict = embed_chunk_strings(file_chunks)
            chunk_dict[file_name] = chunk_embedding_dict
        st.session_state["uploaded_files_embeddings"] = chunk_dict
        update_cache_dict(chunk_dict)
        # st.info(f"Current Session State of uploaded_files_embeddings AFTER ADDING TO SESSION STATE: {st.session_state["uploaded_files_embeddings"]}")
    except Exception as e:
        st.error(e)


def extract_text_from_file(file: UploadedFile) -> str:
    try:
        if (not isinstance(file, UploadedFile)):
            # st.info(file.type)
            raise Exception("Given file is not of type UploadedFile in extract_text_from_file in document_handle.py")
        file_bytes = file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        file_string = ""
        for page in doc:
            file_string = ' '.join([file_string, page.get_text()])
        # st.info(file_string)
        return file_string
    except Exception as e:
        st.error(e)


def create_chunks_from_string(text: str) -> list[str]:
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000,
                                                    chunk_overlap = 80,
                                                    length_function = len,
                                                    is_separator_regex = False)
        doc_chunks = text_splitter.split_text(text)
        return doc_chunks

    except Exception as e:
        st.error(f"Error in create_chunks() in document_handle.py: {e}")
        return []