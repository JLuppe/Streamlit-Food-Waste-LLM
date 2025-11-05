from google import genai
from dotenv import load_dotenv
import streamlit as st

# load environment variables from .env file (api key)
load_dotenv()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

def get_response(conversation, text, context):
    try:
        # file_str = ""
        # if (files):
        #     st.info("not fail 4")
        #     file_str = "Answer the question using the following data as context: " + "\n" + files
        response = client.models.generate_content(
                model="gemini-2.5-flash-lite", contents = (conversation + "User wants to know about" + text +  
                                                           "Using the retrieved data below and the conversation above, provide the best possible answer. Show the final answer clearly. Context: "
                                                             + context + "if no context is relevant, then just answer the question baed on your knowledge.")
        )
        
        # st.info("not fail 5")
        # st.info(conversation)
        return response.text
    except Exception as e:
        st.error(f"API error: {e}")
        return "Fail"