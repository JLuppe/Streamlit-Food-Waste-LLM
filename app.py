import os
import pickle
import streamlit as st
from genAI import get_response
from document_handle import convert_doc
from embedding import rank_chunks_for_question
from streamlit_pdf_viewer import pdf_viewer
from streamlit.runtime.uploaded_file_manager import UploadedFile
import glob

DATA_PATH = "data"

# embedding_cache is a dictionary with chunk string as keys and embeddings as values
EMBEDDING_CACHE_PATH = "permanent_embeddings/embeddings.pkl"

EMBEDDING_CACHE_DIR = "permanent_embeddings"


st.set_page_config(page_title = "Food Waste Insights Tool", layout="wide")
# st.set_page_config(layout="wide")

if "conversation" not in st.session_state:
    st.session_state["conversation"] = ""
if "conversation_list" not in st.session_state:
    st.session_state["conversation_list"] = []
if "API_KEY" not in st.session_state:
    st.session_state["API_KEY"] = ""
if "rag_context" not in st.session_state:
    st.session_state["rag_context"] = ""
if "uploaded_chunks" not in st.session_state:
    st.session_state["uploaded_chunks"] = []
if "chunk_tuples" not in st.session_state:
    st.session_state["chunk_tuples"] = []
if "pdf_binary" not in st.session_state:
    st.session_state["pdf_binary"] = None
if "embedding_cache" not in st.session_state:
    st.session_state["embedding_cache"] = {}

#   FUNCTION:   Initializes pre-computed embeddings via pickle
#   RETURNS:    N/A

def init_embedding_cache():
    pkl_files = glob.glob(os.path.join(EMBEDDING_CACHE_DIR, "*.pkl"))
    if not pkl_files:
        return

    combined_cache = {}

    for path in pkl_files:
        with open(path, "rb") as f:
            data = pickle.load(f)  # dict {chunk_str:, embedding_data}
        combined_cache.update(data)
    st.session_state["embedding_cache"] = combined_cache    

init_embedding_cache()

#   FUNCTION:   Resets conversation, and reinitializes embeddings
#   RETURNS:    N/A
def reset_conversation():
    st.session_state["conversation"] = ""
    st.session_state["conversation_list"] = []
    st.session_state["embedding_cache"] = {}
    st.session_state["rag_context"] = ""
    init_embedding_cache()

#   FUNCTION:   Sets binary data to the pdf of the button that use presses
#   RETURNS:    N/A
def set_ss_binary(file_name: str):
    pattern = os.path.join(DATA_PATH, '**', file_name)
    matches = glob.glob(pattern, recursive=True)
    if matches:
        full_path = matches[0]
        with open(full_path, 'rb') as f:
            binary_data = f.read()
            st.session_state["pdf_binary"] = binary_data

def set_ss_binary_uploaded_file(file: UploadedFile):
    file_binary = file.read()
    st.session_state["pdf_binary"] = file_binary

def generate_uploaded_file_buttons():
    uploadedFiles: list[UploadedFile] = st.session_state["sidebar_uploaded_files"]
    for file in uploadedFiles:
        st.sidebar.button(label= file.name, key=file.name, on_click=set_ss_binary_uploaded_file, args=(file,))

col1, col2 = st.columns(2) 

user_question = st.chat_input("What do you want to know?", width=1000)
with col1:
    st.title("AI Food Waste Insights Tool", width="stretch")
    st.set_page_config(layout="wide")
    chat_container = st.container(height=950)

    if user_question:
        files = []
        with st.spinner("Generating Response..."):
            try:
                if (st.session_state["API_KEY"] != ""):
                    st.session_state["conversation_list"].append(user_question)
                    st.session_state["conversation"] += "\nUser: " + user_question
                    if (st.session_state["uploaded_chunks"] or st.session_state["embedding_cache"]):
                        st.session_state["chunk_tuples"] = rank_chunks_for_question(st.session_state["uploaded_chunks"], user_question)
                        tuples: (list[tuple[str, float]]) = st.session_state["chunk_tuples"] 
                        if (tuples):
                            st.info("Using File Context")
                            for tuple in tuples:
                                st.session_state["rag_context"] = st.session_state["rag_context"] + tuple[0]
                    st.session_state["response"] = get_response(st.session_state["conversation"], user_question, st.session_state["rag_context"])
                    st.session_state["conversation"] += "\nYou: " + st.session_state["response"]
                    st.session_state["conversation_list"].append(st.session_state["response"])
                else:
                    st.info("Please input your API key")

            except Exception as e:
                st.error(f"Something went wrong: {e}")



# Files that come standard
st.sidebar.title("Foundational Knowledge Files")
file_sidebar_container = st.sidebar.container(key="files")
filestr_in_data = os.listdir(DATA_PATH)
for file in filestr_in_data:
    file_sidebar_container.button(file, key=file, on_click=set_ss_binary, args=(file,))

# Files that are uploaded in the user session
st.session_state["API_KEY"] = st.sidebar.text_input("User API Key")
st.sidebar.title("Your Files")
st.session_state["sidebar_uploaded_files"] = st.sidebar.file_uploader("Upload File", accept_multiple_files=True, type="pdf")
if st.session_state["sidebar_uploaded_files"]:
    st.session_state["uploaded_chunks"] = convert_doc(st.session_state["sidebar_uploaded_files"])
    generate_uploaded_file_buttons()

st.sidebar.button("Reset Chat History", on_click = reset_conversation)


# TODOs
# 1. Create buttons to view specific uploaded pdf
with col2:  
    st.title("Document Viewer")
    st.set_page_config(layout="wide")
    if st.session_state["pdf_binary"]:
        pdf_viewer(input=st.session_state["pdf_binary"], height=800, width=800)
    else:
        st.text("No Documents Uploaded")


#   FUNCTION:   prints the conversation
#   RETURNS:    N/A
def print_conversation():
    chat_container.empty()  # Clear all prior messages in this container
    for i in range(len(st.session_state["conversation_list"])):
        with chat_container:
            if (i & 1): # if i is odd
                st.chat_message("assistant").write(st.session_state["conversation_list"][i])
            else:
                st.chat_message("user").write(st.session_state["conversation_list"][i])

print_conversation()