# import packages
import streamlit as st
from rag import handle_files, get_embedding_function, handle_uploaded_files
from genAI import get_response
from langchain_chroma.vectorstores import Chroma
from langchain_classic.document_loaders import PyPDFLoader
import os
from streamlit.runtime.uploaded_file_manager import UploadedFile

DATA_PATH = "data"
CHROMA_PATH = "chroma"

# @st.cache_data

st.title("AI Food Waste Insights", width="stretch")
st.set_page_config(page_title = "Food Waste Insights", layout="centered")

if "conversation" not in st.session_state:
    st.session_state["conversation"] = ""   # record of the entire conversation
if "files" not in st.session_state:
    st.session_state["files"] = []          # files uploaded
if "user_history" not in st.session_state:  
    st.session_state["user_history"] = []   # history of the users questions
if "assisstant_history" not in st.session_state:
    st.session_state["assisstant_history"] = [] # history of the llm responses
if "sources" not in st.session_state:
    st.session_state["sources"] = [] # history of the llm responses

user_question = st.chat_input("What do you want to know?")
chat_container = st.container()

def reset_conversation():
    st.session_state["conversation"] = ""
    st.session_state["user_history"] = []
    st.session_state["assisstant_history"] = []

# def print_conversation():
#      chat_container.empty()
#      for i in range(0, len(st.session_state["user_history"])):
#             st.session_state[i] = st.chat_message("user")
#             st.session_state[i].write(st.session_state["user_history"][i])
#             st.session_state[i] = st.chat_message("assistant")
#             st.session_state[i].write(st.session_state["assisstant_history"][i])

def print_conversation():
    chat_container.empty()  # Clear all prior messages in this container
    for i in range(len(st.session_state["user_history"])):
        with chat_container:
            st.chat_message("user").write(st.session_state["user_history"][i])
            st.chat_message("assistant").write(st.session_state["assisstant_history"][i])

 # convert UploadedFile to Documents to be added to chroma db
def convert_doc(uploaded_files: list[UploadedFile]):
    docs = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        docs.extend(documents)
        os.remove(file.name)
    return docs

if user_question:
    files = []
    try:
        file_str = ""
        handle_files() # create chunks from files in data folder

        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()) # gets the chunk vectors

        results = db.similarity_search_with_score(user_question, k=5) # retrieve the top 5 chunks that are relevant to user_question
        threshold = 340 # threshold where if the similarity score is over (closer to 0 is more similar) the threshold, it won't use that chunk
        filtered_results = [(doc, score) for doc, score in results if score <= threshold]
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results]) # join the chunk content together

        st.session_state["user_history"].append(user_question)
        st.session_state["conversation"] += "\nUser: " + user_question

        response = get_response(st.session_state["conversation"], user_question, context_text)

        sources = [doc.metadata.get("id", None) for doc, _score in filtered_results]
        # for doc in filtered_results:
        #     st.info(doc[0].metadata)            # to see page number, and other metdata 
        

        st.session_state["conversation"] += "\nYou: " + response
        sources_str = ""
        for source in sources:
            sources_str += (" Page #: " + source + "\n")
        st.session_state["assisstant_history"].append(response + "\n" + sources_str)

        # print_conversation()

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.sidebar.title("Foundational Knowledge Files")
file_container = st.sidebar.container(key="files")
filestr_in_data = os.listdir(DATA_PATH)
for file in filestr_in_data:
    # st.button(file, key=i) # TODO: DELETE BUTTON (and files) WHEN CLICKED
    file_container.write(file)
st.sidebar.title("Your Files")

uploaded_files = st.sidebar.file_uploader("Upload File", accept_multiple_files=True)
if uploaded_files:
    uploaded_docs = convert_doc(uploaded_files)
    handle_uploaded_files(uploaded_docs)
    # print_conversation()
st.sidebar.button("Reset Chat History", on_click = reset_conversation)
print_conversation()
