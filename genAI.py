import streamlit as st
from google import genai
import re
import markdown

#   FUNCTION:   Use Gemini API to get response
#   RETURNS:    Response as string
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

1. Use the information in [CONVERSATION SO FAR] and [CONTEXT] to answer [USER QUESTION] as accurately as possible.

2. If the [CONTEXT] section is empty, 'None', 'N/A', or clearly does not contain useful domain information, then:
   - Answer the question based only on your general training data and world knowledge.
   - Explicitly tell the user in your reply that you are *not* using any provided external context and are answering based only on your general knowledge.

3. If relevant information appears in [CONTEXT], prioritize it over your general training data. Do not invent facts that are not supported by either the context or common knowledge.

4. Cite specific chunks from the [CONTEXT] section in your answer that are indicated by "CHUNK ID # X:" where X is the chunk number that contains the supporting information. For example, 
if chunk 2 contains relevant information, include "CHUNK ID # 2:" in your response. If there are no relevant chunks, do not cite any.

5. If the question cannot be fully answered with the conversation and context provided, say what is missing and answer only the part you can justify.

6. RESPOND IN HTML FORMATTED TEXT
""")
        )
        return response.text
    except Exception as e:
        st.error(f"API error, get_response() in genAI.py: {e}")
        return "Fail"


def find_number(chunk_id):
    match = re.search(r"(\d+)", chunk_id)
    if match:
        return match.group(1)
    return None



def add_sources_in_response(response: str, tuples: list[tuple[str, str, str]], chat_container):
    try:

        chunks: list[tuple[str, str]] = []
        
        for i, tup in enumerate(tuples):
            chunk_id = str(i + 1)
            key_str = f"CHUNK ID # {chunk_id}: "
            chunks.append((key_str, tup[1], tup[2])) 
        
        # Build chunk map
        chunk_map = {}
        for key_str, file_name, content in chunks:
            match = re.search(r"(\d+)", key_str)
            if match:
                c_id = match.group(1)
                chunk_map[c_id] = (file_name, content)
        st.session_state["chunk_map"] = chunk_map
        
        # Create HTML with clickable links
        html_content = render_clickable_chunks(response, tuple(chunks))
        return html_content
    except Exception as e:
        st.info("Error in add_sources_in_response in genAI.py: {e}")


def render_clickable_chunks(response: str, chunks: tuple[tuple[str, str]]):
    try:

        if not chunks:
            return response
        
        # First, convert markdown to HTML
        html_content = markdown.markdown(response)
        
        # Then apply link replacements on the HTML
        pattern = r"CHUNK ID # (\d+)"
        
        def make_link(match):
            chunk_id = match.group(1)
            return (
                f"<a href='#' id='{chunk_id}' "
                f"style='background-color: #2e2e2e; padding: 2px 6px; border-radius: 6px; "
                f"color: #ffffff; text-decoration: none; font-weight: bold;'>"
                f"{find_number(chunk_id)}</a>"
            )
        
        html_content = re.sub(pattern, make_link, html_content)
        return html_content
    except Exception as e:
        st.info("Error in render_clickable_chunks in genAI.py: {e}")