import os
import pickle
import streamlit as st
from genAI import get_response
from document_handle import convert_doc
from embedding import rank_chunks_for_question

DATA_PATH = "data"

# embedding_cache is a dictionary with chunk string as keys and embeddings as values
EMBEDDING_CACHE_PATH = "permanent_embeddings/embeddings.pkl"

st.title("AI Food Waste Insights Tool", width="stretch")
st.set_page_config(page_title = "Food Waste Insights Tool", layout="centered")

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

#   FUNCTION:   Initializes pre-computed embeddings via pickle
#   RETURNS:    N/A
def init_embedding_cache():
    if "embedding_cache" not in st.session_state:
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                st.session_state["embedding_cache"] = pickle.load(f)       # pickle files is a dict with {key: chunk str, value: embedding vector}
        else:
            st.session_state["embedding_cache"] = {}

init_embedding_cache()

#   FUNCTION:   Resets conversation, and reinitializes embeddings
#   RETURNS:    N/A
def reset_conversation():
    st.session_state["conversation"] = ""
    st.session_state["conversation_list"] = []
    st.session_state["embedding_cache"] = {}
    init_embedding_cache()

chat_container = st.container()

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
        

user_question = st.chat_input("What do you want to know?")
if user_question:
    files = []
    with st.spinner("Generating Response..."):
        try:
            if (st.session_state["API_KEY"] != ""):
                 st.session_state["conversation_list"].append(user_question)
                 st.session_state["conversation"] += "\nUser: " + user_question
                 if (st.session_state["uploaded_chunks"] or st.session_state["embedding_cache"]):
                     st.session_state["chunk_tuples"] = rank_chunks_for_question(st.session_state["uploaded_chunks"], user_question, 5)
                     tuples: (list[tuple[str, float]]) = st.session_state["chunk_tuples"] 
                     if (tuples):
                        st.info("Using File Context")
                        for tuple in tuples:
                            st.session_state["rag_context"] = st.session_state["rag_context"] + tuple[0]
                 
                 st.session_state["response"] = get_response(st.session_state["conversation"], user_question, st.session_state["rag_context"])           # to see page number, and other metdata 
                 st.session_state["conversation"] += "\nYou: " + st.session_state["response"]
                 st.session_state["conversation_list"].append(st.session_state["response"])
            else:
                st.info("Please input your API key")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Files that come standard
st.sidebar.title("Foundational Knowledge Files")
file_container = st.sidebar.container(key="files")
filestr_in_data = os.listdir(DATA_PATH)
for file in filestr_in_data:
    file_container.write(file)

# Files that are uploaded in the user session
st.session_state["API_KEY"] = st.sidebar.text_input("User API Key")
st.sidebar.title("Your Files")
st.session_state["sidebar_uploaded_files"] = st.sidebar.file_uploader("Upload File", accept_multiple_files=True, type="pdf")
if st.session_state["sidebar_uploaded_files"]:
    st.session_state["uploaded_chunks"] = convert_doc(st.session_state["sidebar_uploaded_files"])
st.sidebar.button("Reset Chat History", on_click = reset_conversation)

print_conversation()