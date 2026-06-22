import streamlit as st
import os
from pypdf import PdfReader
import fitz
import re
from sidebar import set_viewed_file, set_viewed_file_uploaded
from highlight import highlight_by_text_position

DATA_PATH = "data"

#   FUNCTION:   Displays PDF viewer for uploaded or selected document
# def document_viewer():
#     try:
#         css = """
#         .st-key-file_viewer_container {
#             background-color: rgba(255, 255, 255, 1);
#             border-radius: 12px;
#         }
#         """
#         st.title("Document Viewer")
#         file_viewer = st.container(height=950, key="file_viewer_container", border=False)
#         if st.session_state["viewed_file_html"]:
#             col1, col2, col3 = file_viewer.columns([0.1, 0.8, 0.1])
#             file_html = st.session_state["viewed_file_html"]
#             st.html(f"<style>{css}</style>")
#             col2.html(file_html, width="content")
#         else:
#             col1, col2, col3 = file_viewer.columns([0.1, 0.8, 0.1])
#             st.html(f"<style>{css}</style>")
#             col2.html("<p style='text-align: center; color: black; font-weight: bold; font-size: 48px;'>No Document Selected</p>")
            
#             # file_viewer.text("No Documents Selected")
#     except Exception as e:
#         st.info("Error in document_viewer in pdf.py: {e}")

def document_viewer():
    try:
        css = """
        .st-key-file_viewer_container {
            background-color: rgba(255, 255, 255, 1);
            border-radius: 12px;
        }
        """
        st.title("Document Viewer")
        file_viewer = st.container(height=950, key="file_viewer_container", border=False)

        if st.session_state["viewed_file_html"]:
            col1, col2, col3 = file_viewer.columns([0.1, 0.8, 0.1])
            file_html = st.session_state["viewed_file_html"]
            st.html(f"<style>{css}</style>")
            col2.html(file_html, width="content")
        else:
            col1, col2, col3 = file_viewer.columns([0.1, 0.8, 0.1])
            st.html(f"<style>{css}</style>")
            col2.html("""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 870px;
                ">
                    <p style="
                        text-align: center;
                        color: black;
                        font-weight: bold;
                        font-size: 24px;
                        margin: 0;
                    ">No Document Selected</p>
                </div>
            """)
    except Exception as e:
        st.info(f"Error in document_viewer in pdf.py: {e}")
    

#   FUNCTION:  identifies text from all chunks used in context that exist within the provided file name and highlights them in the PDF
#               THIS FUNCTION IS CALLED WHEN THE SOURCE BUTTON IS PRESSED IN THE RESPONSE.
#   MODIFIES:  st.session_state["pdf_binary"]
def filter_and_highlight_foundational_knowledge(file_name: str):
    try:
        with st.spinner("Loading and highlighting foundational knowledge..."):
            pdf_bytes = None
            chunks = []
            
            candidate_path = os.path.join(DATA_PATH, file_name.strip())
            if os.path.isfile(candidate_path):
                with open(candidate_path, "rb") as f:
                    pdf_bytes = f.read()
                set_viewed_file(file_name.strip())
            if (file_name.strip() in [uploaded_file.name for uploaded_file in st.session_state["uploaded_files"]]):
                uploaded_file = next((uf for uf in st.session_state["uploaded_files"] if uf.name == file_name.strip()), None)
                set_viewed_file_uploaded(uploaded_file)
                
            
            original_html = st.session_state["viewed_file_html"]    
            # extracting the chunks that belong to the file
            tuples = st.session_state["chunk_tuples"]
            for tup in tuples:
                if tup[1].strip() == file_name.strip():
                    chunks.append(tup[2])
                    # segments = split_chunk_intelligently(tup[2], min_length=100)
                    # chunks.extend(segments) 
                    
            highlighted_text = highlight_by_text_position(original_html, chunks)

            st.session_state["viewed_file_string"] = highlighted_text
            st.session_state["viewed_file_html"] = highlighted_text
            st.session_state["pdf_binary"] = pdf_bytes
    except Exception as e:
        st.info("Error in filter_and_highlight_foundational_knowledge() in pdf.py: {e}")


def bytes_to_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(pdf_bytes)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.info("Error in bytes_to_text() in pdf.py: {e}")
