from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_aws import BedrockEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_classic.schema import Document
import streamlit as st

DATA_PATH = "data"
CHROMA_PATH = "chroma"

def handle_files():
    documents = load_documents()
    chunks = split_files(documents)
    add_to_chroma(chunks)
    

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_files(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                                   chunk_overlap = 80,
                                                   length_function = len,
                                                   is_separator_regex = False)
    return text_splitter.split_documents(documents)


def add_to_chroma(documents: list[Document]):
    db = Chroma(persist_directory = CHROMA_PATH,
                embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(documents)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []

    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if (len(new_chunks) > 0):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_index += 1
        else:
            current_index = 0
        
        chunk_id = f"{current_page_id}:{current_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    return chunks


def get_embedding_function():
    aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    embeddings = BedrockEmbeddings(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name = "us-east-2",
        model_id="amazon.titan-embed-text-v1"
    )
    return embeddings