import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import pymupdf
import fitz
import difflib
import string
import os

DATA_PATH = "data"

#   FUNCTION:   Displays PDF viewer for uploaded or selected document
def document_viewer():
    st.title("Document Viewer")
    st.set_page_config(layout="wide")
    if st.session_state["pdf_binary"]:
        pdf_viewer(input=st.session_state["pdf_binary"], height=900, width=800)
    else:
        st.text("No Documents Uploaded")

#   FUNCTION:   General function to highlight text in PDF
#   RETURNS:    bytes of highlighted PDF
def highlight_pdf(pdf_bytes: bytes, text_to_highlight: str) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if not doc:
        st.error("Highlighting Error: Could not open PDF document.")
        return pdf_bytes
    translator = str.maketrans('', '', string.punctuation)
    target_words = text_to_highlight.split()
    target_words_clean = [w.translate(translator).lower() for w in target_words]
    
    MIN_CONSECUTIVE_WORDS = 25

    for page in doc:
        # page_words is list of (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        # sort=True means words are in reading order
        page_words = page.get_text("words", sort=True)
        
        page_words_clean = [w[4].translate(translator).lower() for w in page_words]
        matcher = difflib.SequenceMatcher(None, page_words_clean, target_words_clean)
        matches = matcher.get_matching_blocks()
        quads_to_highlight = []
        
        for match in matches:            
            if match.size >= MIN_CONSECUTIVE_WORDS:
                for i in range(match.a, match.a + match.size):
                    # page_words[i] is (x0, y0, x1, y1, "word", ...)
                    word_rect = fitz.Rect(page_words[i][:4])
                    quads_to_highlight.append(word_rect.quad)
        
        if quads_to_highlight:
            annot = page.add_highlight_annot(quads=quads_to_highlight)
            annot.set_colors(stroke=(1, 1, 0))
            annot.update()

    highlighted_pdf_bytes = doc.write()
    doc.close()
    return highlighted_pdf_bytes


# Use chunk tuples, which are list of tuples of file_name, chunk_text that are most similar to current query
# Find file, convert to bytes pass to highlight_pdf along with chunk_text
# Button for source should have function that filters for all tuples that have that file_name, then modifies the pdf_bytes in a loop to highlight all relevant text

def filter_and_highlight_foundational_knowledge(file_name: str):
    with st.spinner("Loading and highlighting foundational knowledge..."):
        pdf_bytes = None

        candidate_path = os.path.join(DATA_PATH, file_name.strip())
        if os.path.isfile(candidate_path):
            with open(candidate_path, "rb") as f:
                pdf_bytes = f.read()
        else:
            st.error("Highlighting Error: File not found.")
            return
        # pdf_bytes should not have the files information, now load current tuples and find relevant chunks and highlight them
        tuples = st.session_state["chunk_tuples"]
        for tup in tuples:
            if tup[1].strip() == file_name.strip():
                chunk_text = tup[2]
                # st.info(chunk_text)
                # with st.spinner("Finding relevant sections in file..."):
                pdf_bytes = highlight_pdf(pdf_bytes, chunk_text)
        st.session_state["pdf_binary"] = pdf_bytes