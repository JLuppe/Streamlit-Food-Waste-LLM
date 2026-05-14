import streamlit as st
from genAI import score_relevance
from sidebar import load_sidebar
st.set_page_config(page_title = "Context Checker", layout="wide")
st.session_state["_uploaded_file_buttons_rendered"] = False
load_sidebar()


def genai_checker():
    if not st.session_state["API_KEY"]:
        st.error("Please enter your API key in the sidebar to use the genAI checker.")
        return
    question = st.session_state["current_user_question"]
    response = st.session_state["response"]
    context = st.session_state["rag_context"]
    if response and context and question:
        with st.spinner("Evaluating response with genAI..."):
            genai_response: str = score_relevance(question, context, response)
            st.session_state["evaluation"] = genai_response

st.button("Evaluate Response", on_click=genai_checker)
checker_box = st.container(height=500)

evaluation = st.session_state.get("evaluation", "")
with checker_box:
    st.markdown(evaluation)

def convert_string_id(id: str) -> str:
    # first 0 - 9, just take 11th position "CHUNK ID # " + str(id) + ": "
    #                                       012345678910      11      12
    if (id[12] == ":"):
        return str(int(id[11])+1)
    else: # its two digits
        return str(int(id[11:13])+1)

col1, col3 = st.columns([0.48, 0.48]) 
with col1:
    st.title("Chunks")
    chunk_container = st.container(height=1000)
    if (not st.session_state["chunk_tuples"]):
        st.error("No Context Loaded")
    elif (len (st.session_state["chunk_tuples"]) > 0):
        for tuple in st.session_state["chunk_tuples"]:
            name = convert_string_id(tuple[0]) + " " + tuple[1]
            chunk_container.expander(name, expanded=False).write(tuple[2])
    else:
        st.error("No context loaded")

with col3:

    st.title("Response")
    response_container = st.container(height = 1000)
    if (not st.session_state["response"]):
        st.error("No response generated yet")
    else:
        response_container.html(st.session_state["response"])
