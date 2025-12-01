from google import genai
from google.genai import types
from langchain_classic.schema import Document
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MAX_BATCH = 25

#   FUNCTION:   Generate embeddings for a list of documents
#   RETURNS:    np.ndarray of embeddings
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
def rank_chunks_for_question(uploaded_chunks: list[Document], question: str, top_k: int = 25) -> list[tuple[str, float]]:
    try:
        # Cached embeddings
        cache: dict[str, np.ndarray] = st.session_state.get("embedding_cache", {})
        # str list from cache dict
        texts: list[str] = []
        # embedding list from cache dict
        emb_list: list[np.ndarray] = []
        # look at each dict entry in cache, add chunk string and embeddings to respective lists
        for text, emb in cache.items(): 
            texts.append(text)
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1 and emb.shape[0] == 3072:
                emb_list.append(emb)
        # updates cache with uploaded files
        update_cache(uploaded_chunks, texts, emb_list, cache)
        # if there are no embeddings, return w/ nothing
        if not emb_list:
            return []
        
        chunk_embeddings = np.vstack(emb_list)
        similar_chunks = get_chunk_similarity(question, chunk_embeddings, top_k, texts)
        return similar_chunks
    
    except Exception as e:
        st.error(f"Error ranking chunks: {e}")
        print(e)
        return []
    
#   FUNCTION:   Calculates uploaded file chunks and adds them to the cache
#   RETURNS:    NONE
def update_cache(chunks: list[Document], texts: list[str], emb_list: list[np.ndarray], cache: dict[str, np.ndarray]):
    if chunks:         
        st.info("Chunks for uploaded files exist in update_cache")     
        chunk_embs = get_embeddings(chunks, "RETRIEVAL_DOCUMENT")
        for doc, emb in zip(chunks, chunk_embs):
            text = doc.metadata["page_content"]
            cache[text] = emb
            texts.append(text)
            emb_list.append(emb)

        st.session_state["embedding_cache"] = cache


#   FUNCTION:   Calculates the question embedding and returns the top k closest chunks that match w/ the query
#   RETURNS:    list[tuple[str, float]] where str is chunk text and float is similarity index
def get_chunk_similarity(question: str, chunk_embeddings, top_k: int, texts:list):
        q_emb = get_query_embedding([question], "RETRIEVAL_QUERY").reshape(1, -1)
        sims = cosine_similarity(q_emb, chunk_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        result: list[tuple[str, float]] = []
        for i in top_idx:
            # texts[i] corresponds to emb_list[i]
            result.append((texts[i], float(sims[i])))
        return result