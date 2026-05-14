import streamlit as st
import os
import glob
from streamlit.runtime.uploaded_file_manager import UploadedFile
from uploaded_file_handle import convert_docs_to_dict_chunks
from embedding import init_embedding_cache
from pypdf import PdfReader
import fitz
import traceback

DATA_PATH = "data"
def load_sidebar():
    try:
        st.session_state["API_KEY"] = st.sidebar.text_input("User Gemini API Key", type='password', value = st.session_state["API_KEY"])

        # FOUNDATIONAL FILES
        title_column = st.sidebar.columns([1, 3, 1])
        st.sidebar.title("Foundational Files", width="stretch")
        context_dropdown = st.sidebar.expander("Context Options", expanded=False)

        file_sidebar_container = st.sidebar.container(key="files")
        filestr_in_data = os.listdir(DATA_PATH)
        for file in filestr_in_data:
            spacer_col1, spacer_col2 = context_dropdown.columns([0.7, 0.3])
            spacer_col2.write(" ")
            col1_sidebar, col2_sidebar = context_dropdown.columns([0.7, 0.3])
            col2_sidebar.button("View", key=file + "_button", on_click=set_viewed_file, args=(file,))
            col1_sidebar.checkbox(
                file,
                key=file,
                value=file in st.session_state.get("files_in_context", []),
                on_change=include_in_context,
                args=(file,)
            )
    
        if not st.session_state["API_KEY"]:

            st.sidebar.write("Please enter your Gemini API key to enable file upload and context selection features.")
        else:
            # USER FILES
            title_column = st.sidebar.columns([1.25, 1, 1])
            st.sidebar.title("Your Files")
            st.session_state["sidebar_uploaded_files"] = st.sidebar.file_uploader("Upload File", accept_multiple_files=True, type="pdf")

            if st.session_state["sidebar_uploaded_files"]:
                # st.info(f"Uploaded {len(st.session_state['sidebar_uploaded_files'])} files.")
                existing_names = {f.name for f in st.session_state["uploaded_files"]}
                new_files = [f for f in st.session_state["sidebar_uploaded_files"] if f.name not in existing_names]
                if new_files:
                    st.session_state["uploaded_files"] += new_files
                    # st.info("Adding new files to session state and processing embeddings...")
                    st.session_state["uploaded_files_embeddings"] = convert_docs_to_dict_chunks(st.session_state["uploaded_files"])
                    # init_embedding_cache()
            generate_uploaded_file_buttons()
            # RESET CHAT HISTORY
            st.sidebar.button("Reset Chat History", on_click = reset_conversation)
    except Exception as e:
        st.info(f"Error in load_sidebar in sidebar.py: {traceback.format_exc()}")


#   FUNCTION:   Generates buttons and checkboxes for uploaded files in sidebar expander
# def generate_uploaded_file_buttons():
#     try:
#         uploadedFiles: list[UploadedFile] = st.session_state["uploaded_files"]
#         if not uploadedFiles:
#             # st.info("No uploaded files to display.")
#             return
#         # st.info(f"st.session_state:['sidebar_uploaded_files']: {st.session_state["sidebar_uploaded_files"][0]}")
#         uploaded_files_dropdown = st.sidebar.expander("Uploaded Files Context", expanded=False)
#         uf_col1, uf_col2 = uploaded_files_dropdown.columns([0.7, 0.3])
#         for file in uploadedFiles:
#             uf_col1.checkbox(label= file.name, key=file.name, value=file.name in st.session_state.get("files_in_context", []), on_change=include_in_context, args=(file.name,))
#             uf_col2.button("View", key=file.name + "_button", on_click=set_viewed_file_uploaded, args=(file,))
#     except Exception as e:
#         traceback.print_exc()
#         st.info(f"Error in generate_uploaded_file_buttons in sidebar.py: {traceback.format_exc()}")
# def generate_uploaded_file_buttons():
#     try:
#         # Guard: skip if already rendered this cycle
#         render_key = "_uploaded_file_buttons_rendered"
#         if st.session_state.get(render_key):
#             return
#         st.session_state[render_key] = True

#         uploadedFiles: list[UploadedFile] = st.session_state.get("uploaded_files", [])
#         if not uploadedFiles:
#             return

#         page_key = st.session_state.get("_active_page", "default")
#         uploaded_files_dropdown = st.sidebar.expander("Uploaded Files Context", expanded=False)
#         uf_col1, uf_col2 = uploaded_files_dropdown.columns([0.7, 0.3])

#         for file in uploadedFiles:
#             checkbox_key = f"{page_key}__{file.name}"
#             button_key   = f"{page_key}__{file.name}__button"

#             uf_col1.checkbox(
#                 label=file.name,
#                 key=checkbox_key,
#                 value=file.name in st.session_state.get("files_in_context", []),
#                 on_change=include_in_context,
#                 args=(file.name,)
#             )
#             uf_col2.button(
#                 "View",
#                 key=button_key,
#                 on_click=set_viewed_file_uploaded,
#                 args=(file,)
#             )
#     except Exception as e:
#         traceback.print_exc()
#         st.info(f"Error in generate_uploaded_file_buttons: {traceback.format_exc()}")

def generate_uploaded_file_buttons():
    try:
        render_key = "_uploaded_file_buttons_rendered"
        if st.session_state.get(render_key):
            return
        st.session_state[render_key] = True

        uploadedFiles: list[UploadedFile] = st.session_state.get("uploaded_files", [])
        # st.info(uploadedFiles)
        if not uploadedFiles:
            return

        page_key = st.session_state.get("_active_page", "default")
        uploaded_files_dropdown = st.sidebar.expander("Uploaded Files Context", expanded=False)
        uf_col1, uf_col2 = uploaded_files_dropdown.columns([0.7, 0.3])

        for file in uploadedFiles:
            checkbox_key = f"{page_key}__{file.name}__{st.session_state["uploaded_files_counter"]}"
            st.session_state["uploaded_files_counter"] += 1
            button_key   = f"{page_key}__{file.name}__button{st.session_state["uploaded_files_counter"]}"

            # Remove stale keys before re-rendering to avoid DuplicateWidgetID errors
            for key in (checkbox_key, button_key):
                if key in st.session_state:
                    continue

            uf_col1.checkbox(
                label=file.name,
                key=checkbox_key,
                value=file.name in st.session_state.get("files_in_context", []),
                on_change=include_in_context,
                args=(file.name,)
            )
            uf_col2.button(
                "View",
                key=button_key,
                on_click=set_viewed_file_uploaded,
                args=(file,)
            )
    except Exception as e:
        traceback.print_exc()
        st.info(f"Error in generate_uploaded_file_buttons: {traceback.format_exc()}")


#   FUNCTION:   Sets binary data to the pdf of the button that use presses
#   RETURNS:    N/A
def set_viewed_file(file_name: str):
    try:
        pattern = os.path.join(DATA_PATH, '**', file_name)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            full_path = matches[0]
            file = fitz.open(full_path)
            html_output = ""
            for page in file:
                html_output += page.get_text('xhtml')
            file_html = f'<span style="color: black;">{html_output}</span>'
            st.session_state["viewed_file_html"] = file_html
            with open(full_path, 'rb') as f:

                binary_data = f.read()
                reader = PdfReader(full_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                st.session_state["pdf_binary"] = binary_data
                st.session_state["viewed_file_string"] = text
    except Exception as e:
        st.info(f"Error in set_viewed_file in sidebar.py: {e}")


def set_viewed_file_uploaded(file: UploadedFile):
    try:
        # st.info(f"Bytes: {file.read()}")
        file.seek(0)  # Move pointer back to the start
        fitz_file = fitz.open(stream=file.read(), filetype='pdf')
        file.seek(0)  # Move pointer back again for the next library
        file_html = ""
        text = ""
        for page in fitz_file:
            file_html += page.get_text('xhtml')
            text += page.get_text() + "\n"
        file_str = f'<span style="color: black;">{file_html}</span>'
        st.session_state["viewed_file_html"] = file_str
        st.session_state["viewed_file_string"] = text
    except Exception as e:
        st.info(f"Error in set_viewed_file_uploaded in sidebar.py: {e}")

#   FUNCTION:   Resets conversation, and reinitializes embeddings
#   RETURNS:    N/A
def reset_conversation():
    try:
        st.session_state["response"] = ""
        st.session_state["chunk_tuples"] = []
        st.session_state["conversation"] = ""
        st.session_state["conversation_list"] = []
        st.session_state["embedding_cache"] = {}
        st.session_state["rag_context"] = ""
        st.session_state["rag_sources"] = []
        st.session_state["chat_container"].empty()
        init_embedding_cache()
    except Exception as e:
        st.info("Error in reset_conversation in sidebar.py: {e}")

def include_in_context(file_name: str):
    try:
        # value = st.session_state[file_name]
        # if value:
        #     st.session_state["files_in_context"].append(file_name)
        if file_name in st.session_state["files_in_context"]:
            st.session_state["files_in_context"].remove(file_name)
        else:
            st.session_state["files_in_context"].append(file_name)
    except Exception as e:
        st.info("Error in include_in_context in sidebar.py: {e}")