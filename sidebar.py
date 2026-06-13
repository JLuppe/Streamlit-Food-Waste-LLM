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


# ─── Sidebar CSS ─────────────────────────────────────────────────────────────
_SIDEBAR_CSS = """
<style>
/* ── Sidebar shell ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #161718 !important;
    border-right: 1px solid #2a2c2f !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem 2rem !important;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

/* ── Sidebar section titles ─────────────────────────────────────────────── */
[data-testid="stSidebar"] h1 {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: clamp(0.875rem, 0.8rem + 0.35vw, 1rem) !important;
    font-weight: 600 !important;
    color: #8a8986 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 1.25rem 0 0.5rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid #2a2c2f !important;
}

/* ── API key input ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stTextInput {
    margin-bottom: 0.25rem !important;
}

[data-testid="stSidebar"] .stTextInput label {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #8a8986 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.35rem !important;
}

[data-testid="stSidebar"] .stTextInput input {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    background: #1c1d1f !important;
    border: 1px solid #2a2c2f !important;
    border-radius: 0.5rem !important;
    color: #e2e1de !important;
    padding: 0.5rem 0.75rem !important;
    transition: border-color 180ms cubic-bezier(0.16,1,0.3,1),
                box-shadow 180ms cubic-bezier(0.16,1,0.3,1) !important;
}

[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #4f98a3 !important;
    box-shadow: 0 0 0 3px oklch(0.55 0.09 192 / 0.18) !important;
    outline: none !important;
}

[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #555452 !important;
}

/* ── Hint / info text ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stText,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown p {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    color: #555452 !important;
    line-height: 1.55 !important;
}

/* ── Expanders ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stExpander {
    background: #1c1d1f !important;
    border: 1px solid #2a2c2f !important;
    border-radius: 0.625rem !important;
    margin-bottom: 0.5rem !important;
    overflow: hidden !important;
}

[data-testid="stSidebar"] .stExpander summary,
[data-testid="stSidebar"] .stExpander [data-testid="stExpanderToggleIcon"],
[data-testid="stSidebar"] details summary {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    color: #8a8986 !important;
    padding: 0.6rem 0.75rem !important;
    transition: color 180ms cubic-bezier(0.16,1,0.3,1) !important;
}

[data-testid="stSidebar"] details summary:hover {
    color: #e2e1de !important;
}

[data-testid="stSidebar"] details[open] summary {
    color: #4f98a3 !important;
    border-bottom: 1px solid #2a2c2f !important;
}

/* ── Checkboxes ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stCheckbox {
    gap: 0.4rem !important;
    padding: 0.25rem 0 !important;
}

[data-testid="stSidebar"] .stCheckbox label {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    color: #e2e1de !important;
    cursor: pointer !important;
    transition: color 180ms ease !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 140px;
}

[data-testid="stSidebar"] .stCheckbox label:hover {
    color: #4f98a3 !important;
}

/* Custom checkbox track */
[data-testid="stSidebar"] .stCheckbox [data-testid="stCheckboxWidget"] > span {
    border-color: #3a3c3f !important;
    background: #1c1d1f !important;
    border-radius: 0.25rem !important;
    transition: background 180ms ease, border-color 180ms ease !important;
}

[data-testid="stSidebar"] .stCheckbox input:checked ~ span {
    background: #4f98a3 !important;
    border-color: #4f98a3 !important;
}

/* ── Sidebar buttons ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    background: #222326 !important;
    color: #8a8986 !important;
    border: 1px solid #2a2c2f !important;
    border-radius: 0.375rem !important;
    padding: 0.3rem 0.6rem !important;
    width: 100% !important;
    transition: background 180ms cubic-bezier(0.16,1,0.3,1),
                color 180ms cubic-bezier(0.16,1,0.3,1),
                border-color 180ms cubic-bezier(0.16,1,0.3,1) !important;
    cursor: pointer !important;
    white-space: nowrap;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #2d3334 !important;
    color: #4f98a3 !important;
    border-color: #4f98a3 !important;
}

[data-testid="stSidebar"] .stButton > button:active {
    background: #253132 !important;
    transform: translateY(1px) !important;
}

/* Reset Chat History — destructive action, use error accent */
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:last-of-type,
[data-testid="stSidebar"] .stButton:last-child > button {
    color: #d163a7 !important;
    border-color: oklch(0.6 0.18 330 / 0.3) !important;
    background: oklch(0.6 0.18 330 / 0.06) !important;
}

[data-testid="stSidebar"] .stButton:last-child > button:hover {
    background: oklch(0.6 0.18 330 / 0.14) !important;
    border-color: #d163a7 !important;
    color: #e67fbf !important;
}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
    background: #1c1d1f !important;
    border: 2px dashed #2a2c2f !important;
    border-radius: 0.625rem !important;
    padding: 0.75rem !important;
    transition: border-color 180ms ease, background 180ms ease !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"]:hover {
    border-color: #4f98a3 !important;
    background: oklch(0.55 0.09 192 / 0.06) !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] p,
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] small {
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    color: #555452 !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button {
    font-size: 0.75rem !important;
    background: #222326 !important;
    border: 1px solid #3a3c3f !important;
    color: #8a8986 !important;
    border-radius: 0.375rem !important;
    padding: 0.25rem 0.5rem !important;
}

/* Uploaded file chip / tag */
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: #222326 !important;
    border: 1px solid #2a2c2f !important;
    border-radius: 0.375rem !important;
    color: #e2e1de !important;
    font-size: 0.75rem !important;
    padding: 0.25rem 0.5rem !important;
    margin-top: 0.25rem !important;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] hr {
    border-color: #2a2c2f !important;
    margin: 0.75rem 0 !important;
}

/* ── Alert / info inside sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] .stAlert {
    background: #1c1d1f !important;
    border-left: 3px solid #4f98a3 !important;
    border-radius: 0.5rem !important;
    font-size: 0.75rem !important;
    color: #8a8986 !important;
    padding: 0.5rem 0.75rem !important;
}

/* ── Row spacers between file entries ───────────────────────────────────── */
[data-testid="stSidebar"] .stVerticalBlock {
    gap: 0.1rem !important;
}
</style>
"""


def _inject_sidebar_css():
    """Inject sidebar CSS once per session."""
    if "_sidebar_css_injected" not in st.session_state:
        st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
        st.session_state["_sidebar_css_injected"] = True


# ─── Main sidebar loader ──────────────────────────────────────────────────────
def load_sidebar():
    _inject_sidebar_css()
    try:
        # ── API Key ──────────────────────────────────────────────────────────
        st.session_state["API_KEY"] = st.sidebar.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state["API_KEY"],
            placeholder="AIza…",
        )

        st.sidebar.markdown("<hr>", unsafe_allow_html=True)

        # ── Foundational Files ───────────────────────────────────────────────
        st.sidebar.title("Foundational Files")
        context_dropdown = st.sidebar.expander("Context Options", expanded=False)

        filestr_in_data = os.listdir(DATA_PATH)
        for file in filestr_in_data:
            context_dropdown.write("")           # visual spacer
            col1_sidebar, col2_sidebar = context_dropdown.columns([0.7, 0.3])
            col1_sidebar.checkbox(
                file,
                key=file,
                value=file in st.session_state.get("files_in_context", []),
                on_change=include_in_context,
                args=(file,),
            )
            col2_sidebar.button(
                "View",
                key=file + "_button",
                on_click=set_viewed_file,
                args=(file,),
            )

        if not st.session_state["API_KEY"]:
            st.sidebar.markdown(
                "<p style='font-size:0.78rem;color:#555452;margin-top:0.5rem'>"
                "Enter your Gemini API key above to enable file upload and context selection."
                "</p>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown("<hr>", unsafe_allow_html=True)

            # ── User Files ───────────────────────────────────────────────────
            st.sidebar.title("Your Files")
            st.session_state["sidebar_uploaded_files"] = st.sidebar.file_uploader(
                "Upload PDFs",
                accept_multiple_files=True,
                type="pdf",
                label_visibility="collapsed",
            )

            if st.session_state["sidebar_uploaded_files"]:
                existing_names = {f.name for f in st.session_state["uploaded_files"]}
                new_files = [
                    f for f in st.session_state["sidebar_uploaded_files"]
                    if f.name not in existing_names
                ]
                if new_files:
                    st.session_state["uploaded_files"] += new_files
                    st.session_state["uploaded_files_embeddings"] = convert_docs_to_dict_chunks(
                        st.session_state["uploaded_files"]
                    )

            generate_uploaded_file_buttons()

            st.sidebar.markdown("<hr>", unsafe_allow_html=True)

            # ── Reset Chat ───────────────────────────────────────────────────
            st.sidebar.button(
                "🗑  Reset Chat History",
                on_click=reset_conversation,
                use_container_width=True,
            )

    except Exception as e:
        st.sidebar.info(f"Error in load_sidebar: {traceback.format_exc()}")



def generate_uploaded_file_buttons():
    try:
        render_key = "_uploaded_file_buttons_rendered"
        if st.session_state.get(render_key):
            return
        st.session_state[render_key] = True

        uploadedFiles: list[UploadedFile] = st.session_state.get("uploaded_files", [])
        if not uploadedFiles:
            return

        page_key = st.session_state.get("_active_page", "default")
        uploaded_files_dropdown = st.sidebar.expander("Uploaded Files Context", expanded=False)

        for file in uploadedFiles:
            checkbox_key = f"{page_key}__{file.name}__{st.session_state['uploaded_files_counter']}"
            st.session_state["uploaded_files_counter"] += 1
            button_key = f"{page_key}__{file.name}__button{st.session_state['uploaded_files_counter']}"

            # ✅ Create a fresh column pair per file row
            uploaded_files_dropdown.write("")  # visual spacer
            uf_col1, uf_col2 = uploaded_files_dropdown.columns([0.7, 0.3])

            uf_col1.checkbox(
                label=file.name,
                key=checkbox_key,
                value=file.name in st.session_state.get("files_in_context", []),
                on_change=include_in_context,
                args=(file.name,),
            )
            uf_col2.button(
                "View",
                key=button_key,
                on_click=set_viewed_file_uploaded,
                args=(file,),
            )
    except Exception as e:
        traceback.print_exc()
        st.sidebar.info(f"Error in generate_uploaded_file_buttons: {traceback.format_exc()}")


# ─── File viewer helpers ──────────────────────────────────────────────────────
def set_viewed_file(file_name: str):
    try:
        pattern = os.path.join(DATA_PATH, "**", file_name)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            full_path = matches[0]
            file = fitz.open(full_path)
            html_output = "".join(page.get_text("xhtml") for page in file)
            st.session_state["viewed_file_html"] = f'<span style="color:black">{html_output}</span>'
            with open(full_path, "rb") as f:
                binary_data = f.read()
            reader = PdfReader(full_path)
            text = "".join(page.extract_text() + "\n" for page in reader.pages)
            st.session_state["pdf_binary"] = binary_data
            st.session_state["viewed_file_string"] = text
    except Exception as e:
        st.sidebar.info(f"Error in set_viewed_file: {e}")


def set_viewed_file_uploaded(file: UploadedFile):
    try:
        file.seek(0)
        fitz_file = fitz.open(stream=file.read(), filetype="pdf")
        file.seek(0)
        html_output = ""
        text = ""
        for page in fitz_file:
            html_output += page.get_text("xhtml")
            text += page.get_text() + "\n"
        st.session_state["viewed_file_html"] = f'<span style="color:black">{html_output}</span>'
        st.session_state["viewed_file_string"] = text
    except Exception as e:
        st.sidebar.info(f"Error in set_viewed_file_uploaded: {e}")


# ─── Conversation / context helpers ──────────────────────────────────────────
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
        st.sidebar.info(f"Error in reset_conversation: {e}")


def include_in_context(file_name: str):
    try:
        if file_name in st.session_state["files_in_context"]:
            st.session_state["files_in_context"].remove(file_name)
        else:
            st.session_state["files_in_context"].append(file_name)
    except Exception as e:
        st.sidebar.info(f"Error in include_in_context: {e}")