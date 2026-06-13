from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import fitz
from embedding import embed_chunk_strings, update_cache_dict
import numpy as np

def convert_docs_to_dict_chunks(uploaded_files: list[UploadedFile]) -> dict[str, dict[str, np.ndarray]]:
    chunk_dict: dict[str, dict[str, np.ndarray]] = {}
    try:
        if not uploaded_files:
            raise Exception("No uploaded files detected in convert_doc_to_dict_chunks in uplpoaded_file_handle.py")
        
        
        for file in uploaded_files:
            file_name = file.name
            # st.info(file_name)
            
            if st.session_state["uploaded_files_embeddings"] and file_name in st.session_state["uploaded_files_embeddings"]:
                # st.info("Name already in embeddings! Will not recalculate Embeddings")
                continue 
            
            file_chunks = create_chunks_from_string(extract_text_from_file(file))
            # st.info(file_name)
            
            chunk_embedding_dict = embed_chunk_strings(file_chunks)
            chunk_dict[file_name] = chunk_embedding_dict
        # st.info(f"chunk_dict: {chunk_dict}")
        if chunk_dict:
            # st.info(f"updating cache dict with {len(chunk_dict)} files worth of chunks and embeddings")
            update_cache_dict(chunk_dict)
            st.session_state["uploaded_files_embeddings"].update(chunk_dict)
        return st.session_state["uploaded_files_embeddings"]  # return the mutated dict itself
    except Exception as e:
        st.error(e)


def extract_text_from_file(file: UploadedFile) -> str:
    try:
        if (not isinstance(file, UploadedFile)):
            # st.info(file.type)
            raise Exception("Given file is not of type UploadedFile in extract_text_from_file in uplpoaded_file_handle.py")
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
        st.error(f"Error in create_chunks() in uplpoaded_file_handle.py: {e}")
        return []
