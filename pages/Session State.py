# Dropdown windows that show the state of each important session state variable
from sidebar import load_sidebar
from App import initialize_session_state
import streamlit as st
initialize_session_state()
st.session_state["_uploaded_file_buttons_rendered"] = False
load_sidebar()
st.set_page_config(page_title = "Session State", layout="wide")
st.title("**!! This page is mainly for debugging purposes !!**")
conversation = st.expander("conversation", expanded=False)
conversation.write(st.session_state["conversation"])

rag_context = st.expander("rag_context", expanded=False)
rag_context.write(st.session_state["rag_context"])

rag_sources = st.expander("rag_sources", expanded=False)
rag_sources.write(st.session_state["rag_sources"])

uploaded_chunks = st.expander("uploaded_chunks", expanded=False)
uploaded_chunks.write(st.session_state["uploaded_chunks"])

chunk_tuples = st.expander("chunk_tuples", expanded=False)
chunk_tuples.write(st.session_state["chunk_tuples"])

embedding_cache = st.expander("embedding_cache", expanded=False)
embedding_cache.write(st.session_state["embedding_cache"])

new_conversation = st.expander("new_conversation", expanded=False)
new_conversation.write(st.session_state["new_conversation"])

files_in_context = st.expander("files_in_context", expanded=False)
files_in_context.write(st.session_state["files_in_context"])

response = st.expander("response", expanded=False)
response.write(st.session_state["response"])

embedding_cache = st.expander("embedding_cache", expanded=False)
embedding_cache.write(st.session_state["embedding_cache"])

viewed_file_string = st.expander("viewed_file_string", expanded=False)
viewed_file_string.write(st.session_state["viewed_file_string"])

document_viewer = st.expander("document_viewer", expanded=False)
document_viewer.write(st.session_state["document_viewer"])

viewed_file_path = st.expander("viewed_file_path", expanded=False)
viewed_file_path.write(st.session_state["viewed_file_path"])

viewed_file_html = st.expander("viewed_file_html", expanded=False)
viewed_file_html.write(st.session_state["viewed_file_html"])

uploaded_files_embeddings = st.expander("uploaded_files_embeddings", expanded=False)
uploaded_files_embeddings.write(st.session_state["uploaded_files_embeddings"])

uploaded_files = st.expander("uploaded_files", expanded=False)
uploaded_files.write(st.session_state["uploaded_files"])

