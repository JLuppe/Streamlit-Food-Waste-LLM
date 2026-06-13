from google import genai
from google.genai import types
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import pickle
import traceback


EMBEDDING_CACHE_DIR = "permanent_embeddings"
MAX_BATCH = 25

#   FUNCTION:   Embed question
#   RETURNS:    np.ndarray of embedding (singular)
def get_query_embedding(question: str, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    try:
        # st.info(question)
        client = genai.Client(api_key=st.session_state["API_KEY"])
        resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[question],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return np.array(resp.embeddings[0].values, dtype=np.float32)
    except Exception as e:
        st.error(f"Error in get_query_embedding() in embedding.py: {traceback.format_exc()}")

#   FUNCTION:   Updates cache, ranks chunks
#   RETURNS:    list[tuple[str, float]] returns top_k number of chunks that are most similar to question
def rank_chunks_for_question(question: str, top_k: int = 25) -> list[tuple[str, str, str]]:
    try:
        files_in_context = st.session_state.get("files_in_context", [])
        full_cache = st.session_state["embedding_cache"]

        cache: dict[str, dict[str, np.ndarray]] = {
            f: full_cache[f] for f in files_in_context if f in full_cache
        }

        if not cache:
            return []

        name_text_tuples: list[tuple[str, str]] = []
        emb_list: list[np.ndarray] = []

        for f_name, subdict in cache.items():
            for chunk_str, emb in subdict.items():
                if isinstance(emb, np.ndarray):
                    arr = emb.astype(np.float32)
                    if arr.ndim == 1 and arr.shape[0] == 3072:
                        emb_list.append(arr)
                        name_text_tuples.append((f_name, chunk_str))

        if not emb_list:
            return []


        chunk_embeddings = np.vstack(emb_list)
        return get_chunk_similarity(question, chunk_embeddings, top_k, name_text_tuples)

    except Exception:
        traceback.print_exc()
        st.error(f"Error in rank_chunks_for_question(): {traceback.format_exc()}")
        return []

def update_cache_dict(entries_to_add: dict[str, dict[str, np.ndarray]]):
    try:
        st.session_state["embedding_cache"].update(entries_to_add)
    except Exception as e:
        st.error(f"Updating cache with a dict failed: {traceback.format_exc()}")


#   FUNCTION:   Calculates the question embedding and returns the top k closest chunks that match w/ the query
#   RETURNS:    list[tuple[str, float]] where str is chunk text and float is similarity index
def get_chunk_similarity(question: str, chunk_embeddings, top_k: int, name_text_tuples) -> list[tuple[str, str, str]]:
    try:
        q_emb = get_query_embedding(question, "RETRIEVAL_QUERY").reshape(1, -1)
        sims = cosine_similarity(q_emb, chunk_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        result: list[tuple[str, str, str]] = []
        for id, i in enumerate(top_idx):
            id_str = "CHUNK ID # " + str(id) + ": "
            result.append((id_str, name_text_tuples[i][0], name_text_tuples[i][1]))
        # st.info(result)
        return result
    except Exception as e:
        st.error(f"Error in get_chunk_similarity() in embedding.py: {traceback.format_exc()}")


#   FUNCTION:   Initializes pre-computed embeddings via pickle
#   RETURNS:    N/A
def init_embedding_cache():
    try:
        files_in_context = st.session_state.get("files_in_context", [])
        if not files_in_context:
            st.session_state["embedding_cache"] = {}
            return

        # Normalize to basenames for consistent matching

        # Normalize to basenames for consistent matching
        desired_names = {os.path.basename(p) for p in files_in_context}
        combined_cache = {}

        pkl_files = glob.glob(os.path.join(EMBEDDING_CACHE_DIR, "*.pkl"))
        combined_cache = {}

        for path in pkl_files:
            with open(path, "rb") as f:
                data = pickle.load(f)
            for fname, subdict in data.items():
                if os.path.basename(fname) in desired_names:
                    combined_cache[os.path.basename(fname)] = subdict

        # Merge uploaded file embeddings
        for file_name, embeddings in st.session_state.get("uploaded_files_embeddings", {}).items():
            base = os.path.basename(file_name)
            if base in desired_names:
                combined_cache[base] = embeddings

                if os.path.basename(fname) in desired_names:
                    combined_cache[os.path.basename(fname)] = subdict


        st.session_state["embedding_cache"] = combined_cache
    except Exception as e:
        st.error(f"Error in init_embedding_cache(): {traceback.format_exc()}")

def embed_chunk_strings(strings: list[str]) -> dict[str, np.ndarray]:
    try:
        dict_str_embed: dict[str, np.ndarray] = {}
        client = genai.Client(api_key=st.session_state["API_KEY"])
        resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=strings,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        for idx, string in enumerate(strings):
            dict_str_embed[string] = np.array(
                resp.embeddings[idx].values, dtype=np.float32
            )
        
        return dict_str_embed
    except Exception as e:
        st.error(f"Problem with embed_chunk_strings() in embeddings.py: {e}")
