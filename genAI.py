import streamlit as st
from google import genai
from google.genai import types
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










