import os
import tempfile
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from langchain_classic.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.document_loaders import PyPDFLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile

def get_response(conversation, text, context):
    client = genai.Client(api_key=st.session_state["API_KEY"])
    try:
        response = client.models.generate_content(
                model="gemini-2.5-flash-lite", contents = (f"""
You are a helpful, truthful assistant.

[CONVERSATION SO FAR]
{conversation}

[USER QUESTION]
{text}

[CONTEXT]
{context}

[INSTRUCTIONS]

1. Use the information in [CONVERSATION SO FAR] and [CONTEXT] to answer [USER QUESTION] as accurately and concisely as possible.
2. If the [CONTEXT] section is empty, 'None', 'N/A', or clearly does not contain useful domain information, then:
   - Answer the question based only on your general training data and world knowledge.
   - Explicitly tell the user in your reply that you are *not* using any provided external context and are answering based only on your general knowledge.
3. If relevant information appears in [CONTEXT], prioritize it over your general training data. Do not invent facts that are not supported by either the context or common knowledge.
4. If the question cannot be fully answered with the conversation and context provided, say what is missing and answer only the part you can justify.
""")
        )
        return response.text
    except Exception as e:
        st.error(f"API error: {e}")
        return "Fail"

MAX_BATCH = 25

def get_embeddings(chunks: list[str], task_type: str) -> np.ndarray:
    client = genai.Client(api_key=st.session_state["API_KEY"])
    all_embs: list[np.ndarray] = []

    for start in range(0, len(chunks), MAX_BATCH):
        batch = chunks[start:start + MAX_BATCH]
        resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        all_embs.extend(np.array(e.values) for e in resp.embeddings)

    return np.vstack(all_embs)


def rank_chunks_for_question(chunks: list[str], question: str, top_k: int = 5) -> list[tuple[str, float]]:
    try:
        chunk_embeddings = get_embeddings(chunks, "RETRIEVAL_DOCUMENT")
        q_emb = get_embeddings([question], "RETRIEVAL_QUERY")[0].reshape(1, -1)

        sims = cosine_similarity(q_emb, chunk_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]

        result: list[tuple[str, float]] = []
        for i in top_idx:
            pair = (chunks[i], float(sims[i]))
            result.append(pair)
            st.info(pair[0])   # log the chunk text (or f"{pair[1]:.3f} – {pair[0][:200]}")
        return result
    except Exception as e:
        st.error(f"Error ranking chunks: {e}")
        print(e)
        return []
        
def create_chunks(documents: list[Document]) -> list[str]:
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                                   chunk_overlap = 80,
                                                   length_function = len,
                                                   is_separator_regex = False)
    doc_chunks = text_splitter.split_documents(documents)
    chunk_list = []
    for chunk in doc_chunks:
        chunk_list.append(chunk.page_content)
    st.info("Chunk list size: " + str(len(chunk_list)))
    return chunk_list

def convert_doc(uploaded_files: list[UploadedFile]):
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(file.getbuffer())
            tmp_filename = tmpfile.name

        loader = PyPDFLoader(tmp_filename)
        documents = loader.load()
        docs.extend(documents)

        os.remove(tmp_filename)

    return create_chunks(docs)



