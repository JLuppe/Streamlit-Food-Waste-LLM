import os
import tempfile
from langchain_classic.schema import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.document_loaders import PyPDFLoader 

#   FUNCTION:   Splits the page_content in a document into chunks and creates a new list of documents that have been split
#   RETUNRNS:   List of Documents (Documents are chunks with their own metadata)
def create_chunks(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000,
                                                   chunk_overlap = 80,
                                                   length_function = len,
                                                   is_separator_regex = False)
    doc_chunks = text_splitter.split_documents(documents)
    for idx, chunk in enumerate(doc_chunks):
        chunk.metadata["page_content"] = chunk.page_content
    return doc_chunks


#   FUNCTION:   Converts a list of UploadedFiles into a list of Documents 
#   RETURNS:    List of Documents
def convert_doc(uploaded_files: list[UploadedFile]) -> list[Document]:
    docs: list[Document] = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(file.getbuffer())
            tmp_filename = tmpfile.name

        loader = PyPDFLoader(tmp_filename)
        documents = loader.load()
        docs.extend(documents)
        os.remove(tmp_filename)
    return create_chunks(docs)


#   REFERENCE PLAN
#   Need to preview files in app first
#   Find chunk str within the files entire body of text -> use title metadata to identify document, then page_content