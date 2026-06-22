import streamlit as st
from genAI import score_relevance
from sidebar import load_sidebar
st.set_page_config(page_title="Context Checker", layout="wide")
st.session_state["_uploaded_file_buttons_rendered"] = False
load_sidebar(False)
# At the top of your page, before other elements
st.markdown("<div style='margin-top: 3rem'></div>", unsafe_allow_html=True)
# ─── Custom CSS Injection ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Design Tokens ──────────────────────────────────────────────────────── */
:root {
  --font-body: 'Satoshi', 'Inter', sans-serif;

  --color-bg:              #0f1011;
  --color-surface:         #161718;
  --color-surface-2:       #1c1d1f;
  --color-surface-offset:  #222326;
  --color-border:          #2a2c2f;
  --color-divider:         #252729;

  --color-text:            #e2e1de;
  --color-text-muted:      #8a8986;
  --color-text-faint:      #555452;

  --color-primary:         #4f98a3;
  --color-primary-hover:   #3d8590;
  --color-primary-active:  #2d6e78;
  --color-primary-glow:    oklch(0.55 0.09 192 / 0.18);
  --color-primary-subtle:  oklch(0.55 0.09 192 / 0.08);

  --color-success:         #6daa45;
  --color-error:           #d163a7;

  --space-1: 0.25rem; --space-2: 0.5rem;  --space-3: 0.75rem;
  --space-4: 1rem;    --space-6: 1.5rem;  --space-8: 2rem;

  --radius-sm: 0.375rem; --radius-md: 0.5rem;
  --radius-lg: 0.75rem;  --radius-xl: 1rem;

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

[data-testid="stAppViewContainer"] {
  background: var(--color-bg) !important;
}

[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: 1px solid var(--color-border);
}

.block-container {
  padding: var(--space-8) var(--space-6) var(--space-4) var(--space-8) !important;
  max-width: 100% !important;
}

/* ── Page titles ────────────────────────────────────────────────────────── */
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

/* ── Containers ─────────────────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"],
.stContainer {
  border-radius: var(--radius-xl) !important;
  border: 1px solid var(--color-border) !important;
  background: var(--color-surface) !important;
}

/* ── Text inputs ────────────────────────────────────────────────────────── */
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

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--color-surface-2) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-lg) !important;
  margin-bottom: var(--space-2) !important;
  overflow: hidden !important;
  transition: border-color var(--transition), box-shadow var(--transition) !important;
}

[data-testid="stExpander"]:hover {
  border-color: var(--color-primary) !important;
  box-shadow: 0 0 0 1px var(--color-primary-glow) !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
  color: var(--color-text-muted) !important;
  font-size: var(--text-sm) !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em !important;
  padding: var(--space-3) var(--space-4) !important;
  background: transparent !important;
  border-bottom: 1px solid transparent !important;
  transition: color var(--transition), border-color var(--transition) !important;
}

[data-testid="stExpander"][open] summary,
[data-testid="stExpander"][open] [data-testid="stExpanderToggleIcon"] {
  color: var(--color-text) !important;
  border-bottom-color: var(--color-divider) !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  padding: var(--space-4) !important;
  font-size: var(--text-sm) !important;
  color: var(--color-text-muted) !important;
  line-height: 1.7 !important;
}

/* ── Columns gap ────────────────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {
  gap: var(--space-4) !important;
  align-items: stretch !important;
}

/* ── Evaluate button — accent primary style ─────────────────────────────── */
[data-testid="stHorizontalBlock"] > div:first-child .stButton > button,
.stButton > button[kind="primary"] {
  background: var(--color-primary) !important;
  border-color: var(--color-primary) !important;
  color: #fff !important;
  font-weight: 600 !important;
  padding: var(--space-2) var(--space-6) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-md) !important;
}

[data-testid="stHorizontalBlock"] > div:first-child .stButton > button:hover,
.stButton > button[kind="primary"]:hover {
  background: var(--color-primary-hover) !important;
  border-color: var(--color-primary-hover) !important;
  box-shadow: 0 0 0 3px var(--color-primary-glow), var(--shadow-md) !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--color-surface); }
::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: 9999px;
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

/* ── Satoshi font ───────────────────────────────────────────────────────── */
@import url('https://api.fontshare.com/v2/css?f[]=satoshi@300,400,500,700&display=swap');
</style>
""", unsafe_allow_html=True)


def genai_checker():
    if not st.session_state["API_KEY"]:
        st.error("Please enter your API key in the sidebar to use the genAI checker.")
        return
    question = st.session_state["current_user_question"]
    response = st.session_state["response"]
    context = st.session_state["rag_context"]
    if response and context and question:
        with st.spinner("Evaluating response with genAI..."):
            genai_response: str = score_relevance(question, context, response)
            st.session_state["evaluation"] = genai_response


st.button("Evaluate Response", on_click=genai_checker)
checker_box = st.container(height=500)

evaluation = st.session_state.get("evaluation", "")
with checker_box:
    st.markdown(evaluation)


def convert_string_id(id: str) -> str:
    # first 0 - 9, just take 11th position "CHUNK ID # " + str(id) + ": "
    #                                       012345678910      11      12
    if (id[12] == ":"):
        return str(int(id[11]) + 1)
    else:  # its two digits
        return str(int(id[11:13]) + 1)


col1, col3 = st.columns([0.48, 0.48])
with col1:
    st.title("Chunks")
    chunk_container = st.container(height=1000)
    if not st.session_state["chunk_tuples"]:
        st.error("No Context Loaded")
    elif len(st.session_state["chunk_tuples"]) > 0:
        for tuple in st.session_state["chunk_tuples"]:
            name = convert_string_id(tuple[0]) + " " + tuple[1]
            chunk_container.expander(name, expanded=False).write(tuple[2])
    else:
        st.error("No context loaded")


with col3:
    st.title("Response")
    response_container = st.container(height=1000)
    if not st.session_state["response"]:
        st.error("No response generated yet")
    else:
        response_container.html(st.session_state["response"])