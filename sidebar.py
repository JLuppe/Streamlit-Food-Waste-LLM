import streamlit as st
import os
import glob
from streamlit.runtime.uploaded_file_manager import UploadedFile
from document_handle import convert_doc
from embedding import init_embedding_cache

DATA_PATH = "data"

def load_sidebar():
    st.session_state["API_KEY"] = st.sidebar.text_input("User Gemini API Key")

    # FOUNDATIONAL FILES
    title_column = st.sidebar.columns([1, 3, 1])
    title_column[1].title("Foundational Knowledge", width="stretch")
    file_sidebar_container = st.sidebar.container(key="files")
    filestr_in_data = os.listdir(DATA_PATH)
    for file in filestr_in_data:
        spacer_col1, spacer_col2 = file_sidebar_container.columns([0.7, 0.3])
        spacer_col2.write(" ") 
        col1_sidebar, col2_sidebar = file_sidebar_container.columns([0.7, 0.3])
        col2_sidebar.button("View", key=file + "_button", on_click=set_pdf_binary, args=(file,))
        col1_sidebar.checkbox(file, key=file, on_change=include_in_context, args=(file,))


    # USER FILES
    title_column = st.sidebar.columns([1.25, 1, 1])
    title_column[1].title("Your Files")
    st.session_state["sidebar_uploaded_files"] = st.sidebar.file_uploader("Upload File", accept_multiple_files=True, type="pdf")
    if st.session_state["sidebar_uploaded_files"]:
        st.session_state["uploaded_chunks"] = convert_doc(st.session_state["sidebar_uploaded_files"])
        generate_uploaded_file_buttons()

    # RESET CHAT HISTORY
    st.sidebar.button("Reset Chat History", on_click = reset_conversation)  



def generate_uploaded_file_buttons():
    uploadedFiles: list[UploadedFile] = st.session_state["sidebar_uploaded_files"]
    for file in uploadedFiles:
        st.sidebar.checkbox(label= file.name, key=file.name, on_change=set_pdf_binary_uf, args=(file,))


#   FUNCTION:   Sets binary data to the pdf of the button that use presses
#   RETURNS:    N/A
def set_pdf_binary(file_name: str):
    pattern = os.path.join(DATA_PATH, '**', file_name)
    matches = glob.glob(pattern, recursive=True)
    if matches:
        full_path = matches[0]
        with open(full_path, 'rb') as f:
            binary_data = f.read()
            st.session_state["pdf_binary"] = binary_data

def set_pdf_binary_uf(file: UploadedFile):
    file_binary = file.read()
    st.session_state["pdf_binary"] = file_binary

#   FUNCTION:   Resets conversation, and reinitializes embeddings
#   RETURNS:    N/A
def reset_conversation():
    st.session_state["response"] = ""
    st.session_state["chunk_tuples"] = []
    st.session_state["conversation"] = ""
    st.session_state["conversation_list"] = []
    st.session_state["embedding_cache"] = {}
    st.session_state["rag_context"] = ""
    st.session_state["rag_sources"] = []
    st.session_state["chat_container"].empty()
    init_embedding_cache()

def include_in_context(file_name: str):
    value = st.session_state[file_name]
    if value:
        st.session_state["files_in_context"].append(file_name)
    if file_name in st.session_state["files_in_context"] and not value:
        st.session_state["files_in_context"].remove(file_name)
        # st.info("removed " + file_name + " from context")
    # init_embedding_cache()