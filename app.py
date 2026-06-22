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


st.set_page_config(
    page_title="Food Waste Insights Tool",
    layout="wide",
    page_icon="🌱",
    menu_items={"Get Help": None, "Report a bug": None, "About": None}
)
st.markdown("<div style='margin-top: 3rem'></div>", unsafe_allow_html=True)
st.markdown("""
<style>

/* ── Design Tokens ──────────────────────────────────────────────────────── */
:root {
  --font-body: 'Satoshi', 'Inter', sans-serif;

  /* Surfaces */
  --color-bg:              #0f1011;
  --color-surface:         #161718;
  --color-surface-2:       #1c1d1f;
  --color-surface-offset:  #222326;
  --color-border:          #2a2c2f;
  --color-divider:         #252729;

  /* Text */
  --color-text:            #e2e1de;
  --color-text-muted:      #8a8986;
  --color-text-faint:      #555452;

  /* Accent — teal */
  --color-primary:         #4f98a3;
  --color-primary-hover:   #3d8590;
  --color-primary-active:  #2d6e78;
  --color-primary-glow:    oklch(0.55 0.09 192 / 0.18);
  --color-primary-subtle:  oklch(0.55 0.09 192 / 0.08);

  /* Success */
  --color-success:         #6daa45;
  --color-error:           #d163a7;

  /* Spacing */
  --space-1: 0.25rem; --space-2: 0.5rem;  --space-3: 0.75rem;
  --space-4: 1rem;    --space-6: 1.5rem;  --space-8: 2rem;

  /* Radius */
  --radius-sm: 0.375rem; --radius-md: 0.5rem;
  --radius-lg: 0.75rem;  --radius-xl: 1rem;

  /* Type */
  --text-xs:   clamp(0.75rem, 0.7rem + 0.25vw, 0.875rem);
  --text-sm:   clamp(0.875rem, 0.8rem + 0.35vw, 1rem);
  --text-base: clamp(1rem, 0.95rem + 0.25vw, 1.125rem);
  --text-lg:   clamp(1.125rem, 1rem + 0.75vw, 1.5rem);
  --text-xl:   clamp(1.5rem, 1.2rem + 1.25vw, 2.25rem);

  --shadow-md: 0 4px 16px oklch(0 0 0 / 0.35);
  --shadow-lg: 0 12px 36px oklch(0 0 0 / 0.45);
  --transition: 180ms cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── Base Reset ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .block-container {
  font-family: var(--font-body) !important;
  background-color: var(--color-bg) !important;
  color: var(--color-text) !important;
}

/* ── App background ─────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
  background: var(--color-bg) !important;
}

[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: 1px solid var(--color-border);
}

/* ── Main block container ───────────────────────────────────────────────── */
.block-container {
  padding: var(--space-6) var(--space-8) !important;
  max-width: 100% !important;
}

/* ── Page title (st.title) ──────────────────────────────────────────────── */
[data-testid="stHeading"] h1,
.stTitle h1 {
  font-family: var(--font-body) !important;
  font-size: var(--text-xl) !important;
  font-weight: 700 !important;
  letter-spacing: -0.025em !important;
  color: var(--color-text) !important;
  margin-bottom: var(--space-4) !important;
  padding-bottom: var(--space-3) !important;
  border-bottom: 1px solid var(--color-border);
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--color-surface) !important;
  border-right: 1px solid var(--color-border) !important;
}

/* Restore Material Icons font — prevents dropdown arrows becoming text */
[data-testid="stSidebar"] [class*="material"],
[data-testid="stSidebar"] [aria-hidden="true"],
.stSelectbox [class*="material"],
.stSelectbox [aria-hidden="true"],
[data-baseweb="select"] [aria-hidden="true"],
[data-baseweb="select"] span[role="presentation"],
[data-baseweb="icon"] {
  font-family: 'Material Icons', 'Material Icons Outlined' !important;
  font-feature-settings: 'liga' !important;
}

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox div {
  background: var(--color-surface-2) !important;
  border: 1px solid var(--color-border) !important;
  color: var(--color-text) !important;
  border-radius: var(--radius-md) !important;
}

/* ── Chat container ─────────────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"],
.stContainer {
  border-radius: var(--radius-xl) !important;
  border: 1px solid var(--color-border) !important;
  background: var(--color-surface) !important;
}

/* ── Chat messages ──────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: var(--space-3) var(--space-4) !important;
  border-radius: var(--radius-lg) !important;
  margin-bottom: var(--space-2) !important;
}

[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage[aria-label*="user"] {
  background: var(--color-primary-subtle) !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has(.stChatMessageContent-user) {
  background: var(--color-primary-subtle) !important;
}

/* Message avatar icons */
[data-testid="stChatMessageAvatarUser"] {
  background: var(--color-primary) !important;
  border-radius: var(--radius-full, 9999px) !important;
  color: #fff !important;
}

[data-testid="stChatMessageAvatarAssistant"] {
  background: var(--color-surface-offset) !important;
  border-radius: var(--radius-full, 9999px) !important;
}

/* ── Chat input ─────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
  background: var(--color-surface-2) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-xl) !important;
  box-shadow: var(--shadow-md) !important;
  padding: var(--space-2) var(--space-4) !important;
  margin-top: var(--space-3) !important;
  transition: border-color var(--transition), box-shadow var(--transition) !important;
}

[data-testid="stChatInput"]:focus-within {
  border-color: var(--color-primary) !important;
  box-shadow: 0 0 0 3px var(--color-primary-glow), var(--shadow-md) !important;
}

[data-testid="stChatInput"] textarea {
  background: transparent !important;
  color: var(--color-text) !important;
  font-family: var(--font-body) !important;
  font-size: var(--text-sm) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
  color: var(--color-text-faint) !important;
}

/* ── Text inputs (API key etc.) ─────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
  background: var(--color-surface-2) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--color-text) !important;
  font-family: var(--font-body) !important;
  font-size: var(--text-sm) !important;
  padding: var(--space-2) var(--space-3) !important;
  transition: border-color var(--transition), box-shadow var(--transition) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--color-primary) !important;
  box-shadow: 0 0 0 3px var(--color-primary-glow) !important;
  outline: none !important;
}

.stTextInput input::placeholder, .stTextArea textarea::placeholder {
  color: var(--color-text-faint) !important;
}

.stTextInput label, .stTextArea label {
  color: var(--color-text-muted) !important;
  font-size: var(--text-xs) !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
  font-family: var(--font-body) !important;
  font-size: var(--text-sm) !important;
  font-weight: 500 !important;
  background: var(--color-surface-offset) !important;
  color: var(--color-text) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-md) !important;
  padding: var(--space-2) var(--space-4) !important;
  transition: background var(--transition), border-color var(--transition), box-shadow var(--transition) !important;
  cursor: pointer !important;
}

.stButton > button:hover {
  background: var(--color-primary) !important;
  border-color: var(--color-primary) !important;
  color: #fff !important;
  box-shadow: 0 0 0 2px var(--color-primary-glow) !important;
}

.stButton > button:active {
  background: var(--color-primary-active) !important;
  transform: translateY(1px) !important;
}

/* ── Selectbox / dropdown ───────────────────────────────────────────────── */
.stSelectbox div[data-baseweb="select"] > div {
  background: var(--color-surface-2) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--color-text) !important;
}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploadDropzone"] {
  background: var(--color-surface-2) !important;
  border: 2px dashed var(--color-border) !important;
  border-radius: var(--radius-lg) !important;
  transition: border-color var(--transition), background var(--transition) !important;
}

[data-testid="stFileUploadDropzone"]:hover {
  border-color: var(--color-primary) !important;
  background: var(--color-primary-subtle) !important;
}

/* ── Info / error / success boxes ───────────────────────────────────────── */
.stAlert {
  background: var(--color-surface-2) !important;
  border-radius: var(--radius-lg) !important;
  border-left: 3px solid var(--color-primary) !important;
  font-family: var(--font-body) !important;
  font-size: var(--text-sm) !important;
  color: var(--color-text) !important;
}

[data-testid="stAlert"][kind="error"],
.stAlert[data-baseweb*="notification"][kind="error"] {
  border-left-color: var(--color-error) !important;
}

[data-testid="stAlert"][kind="success"] {
  border-left-color: var(--color-success) !important;
}

/* ── Spinner ────────────────────────────────────────────────────────────── */
.stSpinner > div {
  border-top-color: var(--color-primary) !important;
}

/* ── Markdown text ──────────────────────────────────────────────────────── */
.stMarkdown, .stMarkdown p, .stWrite p {
  font-family: var(--font-body) !important;
  font-size: var(--text-base) !important;
  color: var(--color-text) !important;
  line-height: 1.7 !important;
}

.stMarkdown a { color: var(--color-primary) !important; }
.stMarkdown code {
  background: var(--color-surface-offset) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.1em 0.35em !important;
  font-size: 0.9em !important;
}

/* ── Columns gap ────────────────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {
  gap: var(--space-4) !important;
  align-items: stretch !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--color-surface); }
::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: var(--radius-full, 9999px);
}
::-webkit-scrollbar-thumb:hover { background: var(--color-text-faint); }

/* ── Source info container ──────────────────────────────────────────────── */
.stContainer[border="true"] {
  background: var(--color-surface-2) !important;
  border-color: var(--color-border) !important;
  border-radius: var(--radius-lg) !important;
}

/* ── Hide Streamlit branding ────────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────────────────────
def initialize_session_state():
    defaults = {
        "response_counter": 0,
        "conversation": "",
        "conversation_list": [],
        "API_KEY": "",
        "rag_context": "",
        "rag_sources": [],
        "uploaded_chunks": [],
        "chunk_tuples": [],
        "pdf_binary": None,
        "embedding_cache": {},
        "new_conversation": [],
        "files_in_context": [],
        "response": "",
        "viewed_file_string": None,
        "document_viewer": None,
        "viewed_file_path": "",
        "viewed_file_html": None,
        "info_box_counter": 100,
        "uploaded_files_embeddings": {},
        "current_user_question": "",
        "evaluation": "None",
        "uploaded_files": [],
        "_uploaded_file_buttons_rendered": False,
        "uploaded_files_counter": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


initialize_session_state()
init_embedding_cache()
st.session_state["_uploaded_file_buttons_rendered"] = False
load_sidebar()


col1, col2 = st.columns(2)
if not st.session_state["API_KEY"]:
    chat_container = st.container(height=1, key="chat_container", vertical_alignment="top", border=False)
    st.markdown("""
    <style>
    .landing-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 72vh;
        padding: var(--space-8) var(--space-4);
        text-align: center;
    }

    .landing-icon {
        font-size: 3.5rem;
        margin-bottom: var(--space-4);
        line-height: 1;
        filter: drop-shadow(0 0 24px oklch(0.55 0.09 192 / 0.35));
    }

    .landing-title {
        font-family: var(--font-body);
        font-size: var(--text-xl);
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--color-text);
        margin-bottom: var(--space-3);
        line-height: 1.15;
    }

    .landing-subtitle {
        font-family: var(--font-body);
        font-size: var(--text-sm);
        color: var(--color-text-muted);
        max-width: 44ch;
        line-height: 1.7;
        margin-bottom: var(--space-8);
    }

    .landing-card {
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-xl);
        padding: var(--space-6) var(--space-8);
        width: 100%;
        max-width: 480px;
        box-shadow: 0 12px 40px oklch(0 0 0 / 0.45);
    }

    .landing-card-label {
        font-family: var(--font-body);
        font-size: var(--text-xs);
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--color-text-muted);
        text-align: left;
    }

    .landing-features {
        display: flex;
        gap: var(--space-3);
        margin-top: var(--space-8);
        flex-wrap: wrap;
        justify-content: center;
        max-width: 520px;
    }

    .landing-feature-chip {
        display: inline-flex;
        align-items: center;
        gap: var(--space-2);
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-full);
        padding: var(--space-2) var(--space-3);
        font-size: var(--text-xs);
        color: var(--color-text-muted);
        font-family: var(--font-body);
    }

    .chip-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--color-primary);
        flex-shrink: 0;
    }
    </style>

    <div class="landing-wrapper">
        <div class="landing-icon">🌱</div>
        <h1 class="landing-title">Food Waste Insights Tool</h1>
        <p class="landing-subtitle">
            Ask questions, explore research, and uncover patterns in food waste data
            powered by RAG and your uploaded documents.
        </p>
        <div class="landing-card">
            <p class="landing-card-label">Get started by entering your API key in the sidebar</p>
        </div>
        <div class="landing-features">
            <span class="landing-feature-chip"><span class="chip-dot"></span>RAG-powered answers</span>
            <span class="landing-feature-chip"><span class="chip-dot"></span>PDF document viewer</span>
            <span class="landing-feature-chip"><span class="chip-dot"></span>Source citations</span>
            <span class="landing-feature-chip"><span class="chip-dot"></span>Semantic search</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:


    # ─── CHAT PANEL ──────────────────────────────────────────────────────────────
    with col1:
        st.title("Chat")
        chat_container = st.container(height=950, key="chat_container", vertical_alignment="top", border=True)
        st.session_state["chat_container"] = chat_container
        user_question = col1.chat_input("Ask something about food waste…", width="stretch")

        if user_question:
            st.session_state["current_user_question"] = user_question
            with st.spinner("Generating response…"):
                try:
                    if st.session_state["API_KEY"]:
                        st.session_state["conversation_list"].append(user_question)
                        st.session_state["conversation"] += "\nUser: " + user_question
                        st.session_state["rag_sources"] = []
                        st.session_state["rag_context"] = ""
                        st.session_state["chunk_tuples"] = []

                        if st.session_state["uploaded_files_embeddings"] or st.session_state["files_in_context"]:
                            st.session_state["chunk_tuples"] = rank_chunks_for_question(user_question)
                            if st.session_state["chunk_tuples"]:
                                for idx, tup in enumerate(st.session_state["chunk_tuples"], start=1):
                                    id_str = f"CHUNK ID # {idx}: "
                                    st.session_state["rag_context"] += id_str + tup[2]
                                    st.session_state["rag_sources"].append(tup[1] + "\n")

                        st.session_state["response"] = get_response(
                            st.session_state["conversation"],
                            user_question,
                            st.session_state["rag_context"],
                        )
                        st.session_state["response_counter"] += 1

                        if st.session_state["chunk_tuples"]:
                            st.session_state["response"] = add_sources_in_response(
                                st.session_state["response"],
                                st.session_state["chunk_tuples"],
                                chat_container,
                            )

                        st.session_state["conversation"] += "\nYou: " + st.session_state["response"]
                        st.session_state["conversation_list"].append(st.session_state["response"])
                    else:
                        st.error("Please enter your API key in the sidebar first.")
                except Exception as e:
                    traceback.print_exc()
                    st.error(f"Something went wrong: {e}")


    # ─── PDF VIEWER PANEL ────────────────────────────────────────────────────────
    with col2:
        document_viewer()


# ─── Helper functions ────────────────────────────────────────────────────────
def handle_source_click(clicked):
    try:
        file_name = st.session_state["chunk_map"].get(clicked)[0]
        chunk_str = st.session_state["chunk_map"].get(clicked)[1]
        if chunk_str:
            info = st.container(border=True, key=st.session_state["info_box_counter"])
            st.session_state["info_box_counter"] += 1
            info.html(
                f"<div style='font-family:var(--font-body,sans-serif);color:var(--color-text,#e2e1de);font-size:0.875rem;line-height:1.6'>"
                f"<p style='font-weight:600;margin-bottom:0.5em;color:var(--color-primary,#4f98a3)'>📎 Context ({clicked})</p>"
                f"{chunk_str}"
                f"<p style='margin-top:0.75em;color:var(--color-text-muted,#8a8986);font-size:0.8rem'>"
                f"<strong>Source:</strong> {file_name}</p></div>"
            )
    except Exception as e:
        st.error(f"Error in handle_source_click(): {e}")


def add_source_buttons_to_container(container):
    files = list(dict.fromkeys(st.session_state["rag_sources"]))

    if not files:
        return

    cols = container.columns(len(files))

    for col, file in zip(cols, files):
        col.button(
            f"📄 {file.strip()}",
            key=f"source_btn_{file}",
            on_click=filter_and_highlight_foundational_knowledge,
            args=(file,),
            use_container_width=True,
        )


def print_conversation():
    try:
        chat_container.empty()
        with chat_container:
            for i, message in enumerate(st.session_state["conversation_list"]):
                if i % 2 == 0:
                    with st.chat_message("user"):
                        st.markdown(message)
                else:
                    with st.chat_message("assistant"):
                        if st.session_state["chunk_tuples"]:
                            temp_key = f"{st.session_state['response_counter']}_{i}"
                            clicked = click_detector(
                                st.session_state["conversation_list"][i],
                                key=temp_key,
                            )
                            if clicked:
                                handle_source_click(clicked)
                        else:
                            st.write(st.session_state["conversation_list"][i])
            add_source_buttons_to_container(chat_container)
    except Exception as e:
        st.error(f"Error in print_conversation(): {e}")


print_conversation()
