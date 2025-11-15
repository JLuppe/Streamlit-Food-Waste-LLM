# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_aws import BedrockEmbeddings
# from langchain_chroma.vectorstores import Chroma
# from langchain_classic.schema import Document
# import streamlit as st
# import boto3
# import tempfile

# DATA_PATH = "data"
# # temp_dir = tempfile.mkdtemp()
# # CHROMA_PATH = temp_dir

# def handle_files():
#     documents = load_documents()
#     chunks = create_chunks(documents)
#     add_to_chroma(chunks)


    







# def add_to_chroma(documents: list[Document]):
#     st.session_state["chroma"] = Chroma(embedding_function=get_embedding_function())
#     chunks_with_ids = calculate_chunk_ids(documents)

#     existing_items = st.session_state["chroma"].get(include=[])
#     existing_ids = set(existing_items["ids"])
#     print(f"Number of existing documents in DB: {len(existing_ids)}")

#     new_chunks = []

#     for chunk in chunks_with_ids:
#         if chunk.metadata["id"] not in existing_ids:
#             new_chunks.append(chunk)

#     if (len(new_chunks) > 0):
#         print(f"Adding new documents: {len(new_chunks)}")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         st.session_state["chroma"].add_documents(new_chunks, ids=new_chunk_ids)
#     else:
#         print("No new documents to add")


# def calculate_chunk_ids(chunks: list[Document]):
#     last_page_id = None
#     current_index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         page = chunk.metadata.get("page")
#         current_page_id = f"{source}:{page}"

#         if current_page_id == last_page_id:
#             current_index += 1
#         else:
#             current_index = 0
        
#         chunk_id = f"{current_page_id}:{current_index}"
#         last_page_id = current_page_id

#         chunk.metadata["id"] = chunk_id
#     return chunks

# bedrock_client = boto3.client(
#     "bedrock-runtime",
#     region_name="us-west-2",
#     aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
#     aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
# )

# def get_embedding_function():
#     embeddings = BedrockEmbeddings(
#     client=bedrock_client,
#     model_id="amazon.titan-embed-text-v1" # INFERENCE MODEL THAT CONTAINS THIS MODEL
#     )
#     return embeddings