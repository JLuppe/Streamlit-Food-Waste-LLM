# import packages
import os
import streamlit as st
from genAI import get_response, convert_doc, rank_chunks_for_question


DATA_PATH = "data"

st.title("AI Food Waste Insights", width="stretch")
st.set_page_config(page_title = "Food Waste Insights Tool", layout="centered")

if "conversation" not in st.session_state:
    st.session_state["conversation"] = ""   # record of the entire conversation
if "conversation_list" not in st.session_state:
    st.session_state["conversation_list"] = []   # record of the entire conversation
if "API_KEY" not in st.session_state:
    st.session_state["API_KEY"] = "" # user api key
if "rag_context" not in st.session_state:
    st.session_state["rag_context"] = ""
if "chunks_str" not in st.session_state:
    st.session_state["chunks_str"] = []             # list of strings
if "chunk_tuples" not in st.session_state:
    st.session_state["chunk_tuples"] = []

def reset_conversation():
    st.session_state["conversation"] = ""
    st.session_state["conversation_list"] = []

chat_container = st.container()
def print_conversation():
    chat_container.empty()  # Clear all prior messages in this container
    for i in range(len(st.session_state["conversation_list"])):
        with chat_container:
            if (i & 1): # if i is odd
                st.chat_message("assistant").write(st.session_state["conversation_list"][i])
            else:
                st.chat_message("user").write(st.session_state["conversation_list"][i])
        
 # convert UploadedFile to Documents to be added to chroma db

user_question = st.chat_input("What do you want to know?")
if user_question:
    files = []
    with st.spinner("Generating Response..."):
        try:
            if (st.session_state["API_KEY"] != ""):
                 st.session_state["conversation_list"].append(user_question)
                 st.session_state["conversation"] += "\nUser: " + user_question
                 if (st.session_state["chunks_str"] != ""):
                     st.session_state["chunk_tuples"] = rank_chunks_for_question(st.session_state["chunks_str"], user_question, 5)
                     tuples: (list[tuple[str, float]]) = st.session_state["chunk_tuples"]      # list of tuples (str, float)
                     if (tuples):
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
st.session_state["sidebar_uploaded_files"] = st.sidebar.file_uploader("Upload File", accept_multiple_files=True)
if st.session_state["sidebar_uploaded_files"]:
    st.session_state["chunks_str"] = convert_doc(st.session_state["sidebar_uploaded_files"])
st.sidebar.button("Reset Chat History", on_click = reset_conversation)

print_conversation()
