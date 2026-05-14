import streamlit as st
from genAI import get_response, add_sources_in_response
from embedding import rank_chunks_for_question
from sidebar import load_sidebar
from embedding import init_embedding_cache
from pdf import document_viewer, filter_and_highlight_foundational_knowledge
from st_click_detector import click_detector
import traceback

DATA_PATH = "data"
EMBEDDING_CACHE_DIR = "permanent_embeddings"

st.set_page_config(page_title = "Food Waste Insights Tool", layout="wide")

# At the top of each page file (e.g., pages/chat.py)
def initialize_session_state():

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
    if "viewed_file_string" not in st.session_state:
            st.session_state["viewed_file_string"] = None
    if "document_viewer" not in st.session_state:
        st.session_state["document_viewer"] = None
    if "viewed_file_path" not  in st.session_state:
        st.session_state["viewed_file_path"] = ""
    if "viewed_file_html" not  in st.session_state:
        st.session_state["viewed_file_html"] = None

    if "info_box_counter" not in st.session_state:
        st.session_state["info_box_counter"] = 100

    if "uploaded_files_embeddings" not in st.session_state:
        st.session_state["uploaded_files_embeddings"] = {}
    if "current_user_question" not in st.session_state:
        st.session_state["current_user_question"] = ""
    if "evaluation" not in st.session_state:
        st.session_state["evaluation"] = "None"
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    if "_uploaded_file_buttons_rendered" not in st.session_state:
        st.session_state["_uploaded_file_buttons_rendered"] = False
    if "uploaded_files_counter" not in st.session_state:
        st.session_state["uploaded_files_counter"] = 0
initialize_session_state()
init_embedding_cache()
st.session_state["_uploaded_file_buttons_rendered"] = False
load_sidebar()


col1, col2 = st.columns(2) 


# CHAT

# search_term = search_col.chat_input("Search for text:", width=925, )
with col1:
    st.title("AI Food Waste Insights Tool", width="stretch")
    chat_container = st.container(height=950, key="chat_container", vertical_alignment="top")
    st.session_state["chat_container"] = chat_container
    user_question = col1.chat_input("What do you want to know?", width="stretch" )   
     # st.session_state["rag_context"] = ""
    if user_question:
        st.session_state["current_user_question"] = user_question
        files = []
        with st.spinner("Generating Response..."):
            try:
                if (st.session_state["API_KEY"] != ""):
                    st.session_state["conversation_list"].append(user_question)
                    st.session_state["conversation"] += "\nUser: " + user_question
                    st.session_state["rag_sources"] = []
                    st.session_state["rag_context"] = ""
                    st.session_state["chunk_tuples"] = []
                    tuples: (list[tuple[str, str, str]]) = st.session_state["chunk_tuples"]
                    if (st.session_state["uploaded_files_embeddings"] or st.session_state["files_in_context"]):
                        # TODO: clean up how you get tuples (change rank_chunks_for_question)
                        st.session_state["chunk_tuples"] = rank_chunks_for_question(user_question)
                        # st.info(f"state of chunk_tuples: {st.session_state['chunk_tuples']}")
                        # TUPLE IS (file_name, chunk_str, chunk_text)
                        if (st.session_state["chunk_tuples"]):
                            # st.info("Found " + str(len(st.session_state["chunk_tuples"])) + " relevant chunks in context for this question.")
                            id: int = 1
                            for tuple in st.session_state["chunk_tuples"]:
                                id_str = "CHUNK ID # " + str(id) + ": "
                                id += 1
                                st.session_state["rag_context"] = st.session_state["rag_context"] + id_str + tuple[2]
                                st.session_state["rag_sources"].append(tuple[1] + "\n")
                    st.session_state["response"] = get_response(st.session_state["conversation"], user_question, st.session_state["rag_context"])
                    st.session_state["response_counter"] += 1
                    if (st.session_state["chunk_tuples"]):
                        st.session_state["response"] = add_sources_in_response(st.session_state["response"], st.session_state["chunk_tuples"], chat_container)

                    st.session_state["conversation"] += "\nYou: " + st.session_state["response"]
                    st.session_state["conversation_list"].append(st.session_state["response"])
                else:
                    st.error("Please input your API key")

            except Exception as e:
                traceback.print_exc()
                st.error(f"Something went wrong: {e}")

# PDF VIEWER
with col2:
    document_viewer()


#   FUNCTION:   When button clicked, write chunk data as HTML into a separate container.
#   RETUNRS:    N/A
def handle_source_click(clicked):
    try:
        file_name = st.session_state["chunk_map"].get(clicked)[0]
        chunk_str = st.session_state["chunk_map"].get(clicked)[1]
        # filter_and_highlight_foundational_knowledge(file_name)
        if chunk_str:
            info = st.container(border=True, key=st.session_state["info_box_counter"])
            st.session_state["info_box_counter"] += 1
            info.html(f"<div style='color: white'><b>Context ({clicked})</b> <br />\n\n {chunk_str} \n\n <br /><b>Source File:</b> {file_name}</div>")
    except Exception as e:
        st.error("Error in handle_source_click() in app.py: {e}")

def add_source_buttons_to_container(container):
    for file in set(st.session_state["rag_sources"]):
        container.button(("Source: " + file), key=file, on_click=filter_and_highlight_foundational_knowledge, args=(file,))

#   FUNCTION:   prints the conversation, and adds source files at the end
#   RETURNS:    N/A
# def print_conversation():
#     try:
#         chat_container.empty()
#         for i in range(len(st.session_state["conversation_list"])):
#             with chat_container:
#                 if i % 2 == 0: # user
#                     with st.chat_message("user"):
#                         st.markdown(st.session_state["conversation_list"][i])
#                 else:
#                     with st.chat_message("assistant"):
#                         if st.session_state["chunk_tuples"]:
#                             temp_key = f"{st.session_state['response_counter']}_{i}"
#                             clicked = click_detector(st.session_state["conversation_list"][i], key=temp_key)
#                             if clicked:
#                                 handle_source_click(clicked)
#                         else:
#                             st.write(st.session_state["conversation_list"][i])
#         add_source_buttons_to_container(chat_container)
#     except Exception as e:
#         st.error("Error in print_conversation() in app.py: {e}")

def print_conversation():
    try:
        st.markdown("""
            <style>
            /* Tighten gap between chat messages */
            .stChatMessage {
                padding-top: 0.4rem !important;
                padding-bottom: 0.4rem !important;
                margin-bottom: 0.25rem !important;
            }

            /* Reduce internal padding of message content */
            .stChatMessage > div {
                gap: 0.5rem !important;
            }
            </style>
            """, unsafe_allow_html=True)
        chat_container.empty()
        with chat_container:  # ← single context outside the loop
            for i in range(len(st.session_state["conversation_list"])):
                if i % 2 == 0:
                    with st.chat_message("user"):
                        st.markdown(st.session_state["conversation_list"][i])
                else:
                    with st.chat_message("assistant"):
                        if st.session_state["chunk_tuples"]:
                            temp_key = f"{st.session_state['response_counter']}_{i}"
                            clicked = click_detector(
                                st.session_state["conversation_list"][i],
                                key=temp_key
                            )
                            if clicked:
                                handle_source_click(clicked)
                        else:
                            st.write(st.session_state["conversation_list"][i])
            add_source_buttons_to_container(chat_container)
    except Exception as e:
        st.error(f"Error in print_conversation() in app.py: {e}")
print_conversation()
