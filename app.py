import streamlit as st
from genAI import get_response, add_sources_in_response
from embedding import rank_chunks_for_question
from sidebar import load_sidebar
from embedding import init_embedding_cache
from pdf import document_viewer, filter_and_highlight_foundational_knowledge
from st_click_detector import click_detector
import markdown

DATA_PATH = "data"
EMBEDDING_CACHE_DIR = "permanent_embeddings"


st.set_page_config(page_title = "Food Waste Insights Tool", layout="wide")

if "response_counter" not in st.session_state:
    st.session_state["response_counter"] = 0
if "conversation" not in st.session_state:
    st.session_state["conversation"] = ""
if "conversation_list" not in st.session_state:
    st.session_state["conversation_list"] = []
if "API_KEY" not in st.session_state:
    st.session_state["API_KEY"] = ""
if "rag_context" not in st.session_state:
    st.session_state["rag_context"] = ""
if "rag_sources" not in st.session_state:
    st.session_state["rag_sources"] = []
if "uploaded_chunks" not in st.session_state:
    st.session_state["uploaded_chunks"] = []
if "chunk_tuples" not in st.session_state:
    st.session_state["chunk_tuples"] = []
if "pdf_binary" not in st.session_state:
    st.session_state["pdf_binary"] = None
if "embedding_cache" not in st.session_state:
    st.session_state["embedding_cache"] = {}
if "new_conversation" not in st.session_state:
    st.session_state["new_conversation"] = []
if "files_in_context" not in st.session_state:
    st.session_state["files_in_context"] = []
if "response" not in st.session_state:
    st.session_state["response"] = ""
# Embedding cache: dict[chunk_str, np.ndarray] -> Want to change to dict[file_name, dict[chunk_str, np.ndarray]]
if "embedding_cache" not in st.session_state:
        st.session_state["embedding_cache"] = {}

init_embedding_cache()
load_sidebar()


col1, col2 = st.columns(2) 
# PDF VIEWER
with col2:
    document_viewer()
# CHAT
user_question = st.chat_input("What do you want to know?", width=925)
with col1:
    st.title("AI Food Waste Insights Tool", width="stretch")
    chat_container = st.container(height=950, key="chat_container")
    st.session_state["chat_container"] = chat_container
    st.session_state["rag_context"] = ""
    if user_question:
        files = []
        with st.spinner("Generating Response..."):
            try:
                if (st.session_state["API_KEY"] != ""):
                    st.session_state["conversation_list"].append(user_question)
                    st.session_state["conversation"] += "\nUser: " + user_question
                    st.session_state["rag_sources"] = []
                    st.session_state["rag_context"] = ""
                    tuples = None
                    st.session_state["chunk_tuples"] = []
                    if (st.session_state["uploaded_chunks"] or st.session_state["files_in_context"]):
                        st.session_state["chunk_tuples"] = rank_chunks_for_question(st.session_state["uploaded_chunks"], user_question)
                        tuples: (list[tuple[str, str, str]]) = st.session_state["chunk_tuples"]
                        # TUPLE IS (file_name, chunk_str, chunk_text)
                        if (tuples):
                            id: int = 1
                            for tuple in tuples:
                                id_str = "CHUNK ID # " + str(id) + ": "
                                id += 1
                                st.session_state["rag_context"] = st.session_state["rag_context"] + id_str + tuple[2]
                                st.session_state["rag_sources"].append(tuple[1] + "\n")
                    st.session_state["response"] = get_response(st.session_state["conversation"], user_question, st.session_state["rag_context"])
                    st.session_state["response_counter"] += 1
                    if (tuples):
                        st.session_state["response"] = add_sources_in_response(st.session_state["response"], st.session_state["chunk_tuples"], chat_container)

                    st.session_state["conversation"] += "\nYou: " + st.session_state["response"]
                    st.session_state["conversation_list"].append(st.session_state["response"])
                else:
                    st.info("Please input your API key")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

#   FUNCTION:   prints the conversation, and adds source files at the end
#   RETURNS:    N/A
def print_conversation():
    chat_container.empty()
    for i in range(len(st.session_state["conversation_list"])):
        with chat_container:
            if i % 2 == 0: # user
                with st.chat_message("user"):
                    st.write(st.session_state["conversation_list"][i])
            else:
                with st.chat_message("assistant"):
                    if st.session_state["files_in_context"]:
                        temp_key = f"{st.session_state['response_counter']}_{i}"
                        clicked = click_detector(st.session_state["conversation_list"][i], key=temp_key)
                        if clicked:
                            file_name = st.session_state["chunk_map"].get(clicked)[0]
                            chunk_str = st.session_state["chunk_map"].get(clicked)[1]
                            if chunk_str:
                                st.info(f"**Context ({clicked})** \n\n {chunk_str} \n\n **Source File:** {file_name}")
                    else:
                        st.write(st.session_state["conversation_list"][i])
    for file in set(st.session_state["rag_sources"]):
        chat_container.button(("Source: " + file), key=file, on_click=filter_and_highlight_foundational_knowledge, args=(file,))

print_conversation()



