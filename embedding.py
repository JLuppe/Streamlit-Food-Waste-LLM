from google import genai
from google.genai import types
from langchain_classic.schema import Document
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import pickle

EMBEDDING_CACHE_DIR = "permanent_embeddings"
MAX_BATCH = 25

#   FUNCTION:   Generate embeddings for a list of documents
#   RETURNS:    np.ndarray of embeddings
def get_embeddings(chunks: list[Document], task_type: str) -> np.ndarray:
    client = genai.Client(api_key=st.session_state["API_KEY"])
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

#   FUNCTION:   Embed question
#   RETURNS:    np.ndarray of embedding (singular)
def get_query_embedding(question: str, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    client = genai.Client(api_key=st.session_state["API_KEY"])
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[question],
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array(resp.embeddings[0].values, dtype=np.float32)

#   FUNCTION:   Updates cache, ranks chunks
#   RETURNS:    list[tuple[str, float]] returns top_k number of chunks that are most similar to question

#   TODO: CACHE IS NOW dict[file_name, dict[chunk_str, np.ndarray]]
#         NEED TO UPDATE THIS FUNCTION TO HANDLE NEW CACHE STRUCTURE
#         Tuple with file name and chunk string
#         
def rank_chunks_for_question(uploaded_chunks: list[Document], question: str, top_k: int = 25) -> list[tuple[str,str, str]]:
    # Cached embeddings
    cache: dict[str, dict[str, np.ndarray]] = {}
    # st.info("Files in context: " + str(st.session_state.get("files_in_context", [])))
    for file_name in st.session_state.get("files_in_context", []):
        cache[file_name] = st.session_state["embedding_cache"].get(file_name, {})

    f_names: list[str] = []
    # str list from cache dict
    texts: list[str] = []
    name_text_tuples: list[tuple[str, str]] = []
    # embedding list from cache dict
    emb_list: list[np.ndarray] = []
    # look at each dict entry in cache, add chunk string and embeddings to respective lists
    for f_name in cache.keys():
        f_names.append(f_name)
        for chunk_str, emb in cache.get(f_name).items():
            name_text_tuples.append((f_name, chunk_str))
            texts.append(chunk_str)
            if isinstance(emb, np.ndarray):
                arr = emb.astype(np.float32)
            if arr.ndim == 1 and arr.shape[0] == 3072:
                emb_list.append(arr)
    # updates cache with uploaded files
    update_cache(uploaded_chunks, texts, emb_list, name_text_tuples, cache)
    # if there are no embeddings, return w/ nothing
    if not emb_list:
        return []
    
    chunk_embeddings = np.vstack(emb_list)
    similar_chunks = get_chunk_similarity(question, chunk_embeddings, top_k, name_text_tuples)
    return similar_chunks
    
#   FUNCTION:   Calculates uploaded file chunks and adds them to the cache
#   RETURNS:    NONE

#   TODO: MAKE FUNCTION ADHERE TO NEW CACHE STRUCTURE dict[file_name, dict[chunk_str, np.ndarray]]
#         To do this,
def update_cache(chunks: list[Document], texts: list[str], emb_list: list[np.ndarray], name_text_tuples: list[tuple], cache: dict[str, dict[str, np.ndarray]]):
    if chunks:         
        chunk_embs = get_embeddings(chunks, "RETRIEVAL_DOCUMENT")
        for doc, emb in zip(chunks, chunk_embs):
            text = doc.metadata["page_content"]
            name = doc.id
            cache[name][text] = emb
            name_text_tuples.append((name, text))
            texts.append(text)
            emb_list.append(emb)

        st.session_state["embedding_cache"] = cache


#   FUNCTION:   Calculates the question embedding and returns the top k closest chunks that match w/ the query
#   RETURNS:    list[tuple[str, float]] where str is chunk text and float is similarity index
def get_chunk_similarity(question: str, chunk_embeddings, top_k: int, name_text_tuples) -> list[tuple[str, str, str]]:
        q_emb = get_query_embedding([question], "RETRIEVAL_QUERY").reshape(1, -1)
        sims = cosine_similarity(q_emb, chunk_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        result: list[tuple[str, str, str]] = []
        id: int = 1
        for i in top_idx:
            id_str = "CHUNK ID # " + str(id) + ": "
            result.append((id_str, name_text_tuples[i][0], name_text_tuples[i][1]))
        return result

#   FUNCTION:   Initializes pre-computed embeddings via pickle
#   RETURNS:    N/A
def init_embedding_cache():
    pkl_files = glob.glob(os.path.join(EMBEDDING_CACHE_DIR, "*.pkl"))
    if not pkl_files:
        return
    combined_cache = {}

    # Only load caches for files listed in session_state["files_in_context"]
    files_in_context = st.session_state.get("files_in_context", [])
    if not files_in_context:
        st.session_state["embedding_cache"] = {}
        return
    desired_names = {os.path.basename(p) for p in files_in_context}

    for path in pkl_files:
        with open(path, "rb") as f:
            data = pickle.load(f)  # expected dict[file_name, dict[chunk_str, np.ndarray]]
        for fname, subdict in data.items():
            if fname in desired_names:
                # st.info(fname + " in checkbox, loading embeddings into cache")
                combined_cache[fname] = subdict
    st.session_state["embedding_cache"] = combined_cache