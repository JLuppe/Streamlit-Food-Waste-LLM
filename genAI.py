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
def get_embeddings(chunks: list[Document], task_type: str) -> np.ndarray:
    client = genai.Client(api_key=st.session_state["API_KEY"])

    if "embedding_cache" not in st.session_state:
        st.session_state["embedding_cache"] = {}
    cache: dict[str, np.ndarray] = st.session_state["embedding_cache"]

    to_embed: list[str] = []
    for doc in chunks:
        page_content = doc.metadata["page_content"]
        if page_content not in cache:
            to_embed.append(page_content)

    for start in range(0, len(to_embed), MAX_BATCH):
        batch = to_embed[start:start + MAX_BATCH]
        if not batch:
            continue

        resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )

        for page_content, e in zip(batch, resp.embeddings):
            cache[page_content] = np.array(e.values, dtype=np.float32)

    st.session_state["embedding_cache"] = cache

    all_embs = [cache[doc.metadata["page_content"]] for doc in chunks]
    return np.vstack(all_embs)


def rank_chunks_for_question(chunksR: list[Document], question: str, top_k: int = 5) -> list[tuple[str, float]]:
    try:
        cache: dict[str, np.ndarray] = st.session_state.get("embedding_cache", {})

        texts: list[str] = []
        emb_list: list[np.ndarray] = [] # list of embeddings

        for text, emb in cache.items(): # look at each dict entry in cache, add chunk string and embeddings to respective lists
            texts.append(text)
            emb_list.append(emb)

        if chunksR:                     
            chunk_embs = get_embeddings(chunksR, "RETRIEVAL_DOCUMENT")  # if there are chunks passed as argument (dynamic), get their embeddings

            for doc, emb in zip(chunksR, chunk_embs):
                text = doc.metadata["page_content"]
                if text not in cache:
                    cache[text] = emb
                    texts.append(text)
                    emb_list.append(emb)

            st.session_state["embedding_cache"] = cache

        if not emb_list:
            return []

        chunk_embeddings = np.vstack(emb_list)  

        q_emb = get_query_embedding([question], "RETRIEVAL_QUERY").reshape(1, -1)
        sims = cosine_similarity(q_emb, chunk_embeddings)[0]

        top_idx = np.argsort(sims)[::-1][:top_k]

        result: list[tuple[str, float]] = []
        for i in top_idx:
            # texts[i] corresponds to emb_list[i]
            result.append((texts[i], float(sims[i])))

        return result

    except Exception as e:
        st.error(f"Error ranking chunks: {e}")
        print(e)
        return []




def get_query_embedding(question: str, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    client = genai.Client(api_key=st.session_state["API_KEY"])
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[question],
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array(resp.embeddings[0].values, dtype=np.float32)




def create_chunks(documents: list[Document]) -> list[Document]:         # list of documents in is split into document chunks (for metadata)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000,
                                                   chunk_overlap = 80,
                                                   length_function = len,
                                                   is_separator_regex = False)
    doc_chunks = text_splitter.split_documents(documents)
    for idx, chunk in enumerate(doc_chunks):
        chunk.metadata["page_content"] = chunk.page_content
    # st.info(doc_chunks[0].metadata)
    return doc_chunks





def convert_doc(uploaded_files: list[UploadedFile]) -> list[Document]:
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



