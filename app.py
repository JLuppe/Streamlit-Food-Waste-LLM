# import packages
import streamlit as st
from rag import handle_files, get_embedding_function
from genAI import get_response
from langchain_chroma.vectorstores import Chroma
import os
DATA_PATH = "data"
CHROMA_PATH = "chroma"

# @st.cache_data

st.title("Food Waste GenAI")
st.set_page_config(layout="wide")

if "conversation" not in st.session_state:
    st.session_state["conversation"] = ""   # record of the entire conversation

if "files" not in st.session_state:
    st.session_state["files"] = ""          # files uploaded

if "user_history" not in st.session_state:  
    st.session_state["user_history"] = []   # history of the users questions

if "assisstant_history" not in st.session_state:
    st.session_state["assisstant_history"] = [] # history of the llm responses

col1, col2 = st.columns(2)

user_question = st.chat_input("What do you want to know?",
                              accept_file=True,
                              file_type=['pdf'])
# with col1:

def reset_conversation():
    st.session_state["conversation"] = ""
    st.session_state["user_history"] = []
    st.session_state["assisstant_history"] = []



if user_question:
    files = []
    if (len(user_question.files) != 0):
        files = user_question.files
        for file in files:
            save_path = file.name
            with open(os.path.join("data", file.name), "wb") as f:
                f.write(file.getvalue())
    user_question = user_question.text
    try:
        file_str = ""
        handle_files() # this will create chunks from all the data in rag_test
                    # and assign it a numerical vector that can be compared
                    # with user_question to retrieve most relevant chunks

        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()) # gets the chunk vectors

        results = db.similarity_search_with_score(user_question, k=5) # retrieve the top 5 chunks that are relevant to user_question
        st.info("Most similar result: " + str(results[0][1]))
        st.info("Second most similar result: " + str(results[1][1]))
        st.info("Third most similar result: " + str(results[2][1]))
        threshold = 340
        filtered_results = [(doc, score) for doc, score in results if score <= threshold]

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results]) # join the chunk content together

        st.session_state["user_history"].append(user_question)
        st.session_state["conversation"] += "\nUser: " + user_question

        response = get_response(st.session_state["conversation"], user_question, context_text)

        sources = [doc.metadata.get("id", None) for doc, _score in filtered_results]

        st.session_state["conversation"] += "\nYou: " + response
        sources_str = ""
        for source in sources:
            sources_str += ("\n" + source)
        st.session_state["assisstant_history"].append(response + "\n" + sources_str)
        

        for i in range(0, len(st.session_state["user_history"])):
            st.session_state[i] = st.chat_message("user")
            st.session_state[i].write(st.session_state["user_history"][i])
            st.session_state[i] = st.chat_message("assistant")
            st.session_state[i].write(st.session_state["assisstant_history"][i])

    except Exception as e:
        st.error(f"Something went wrong: {e}")

with st.sidebar:    
    st.title("Your Files")
    filestr_in_data = os.listdir(DATA_PATH)
    i = 1
    for file in filestr_in_data:
        # st.button(file, key=i) # TODO: DELETE BUTTON (and files) WHEN CLICKED
        st.write(file)
        i += 1
    uploaded_file = st.file_uploader("Upload File")
    if (uploaded_file):
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
    st.button("Reset Chat History", on_click = reset_conversation)


