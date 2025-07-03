import os
import streamlit as st
from dotenv import load_dotenv
from chat_bot import ChatbotApp

if __name__ == "__main__":
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    app = ChatbotApp()
    app.run()
