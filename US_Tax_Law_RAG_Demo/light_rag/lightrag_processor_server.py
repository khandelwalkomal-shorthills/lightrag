import os
import sys
import time # Though less used in API, it's there from your original
from dataclasses import asdict
from dotenv import load_dotenv
# FastAPI and Pydantic imports
import asyncio
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
 
 
 
 
# --- CHANGE THESE LINES ---
 
from pdf_loader import PDFLoader              # <--- Changed
from knowledge_base import KnowledgeBase      # <--- Changed
from _astradb import AstraDBKVStorage, AstraDBVectorStorage # <--- Changed
# --- END CHANGES ---
 
# --- IMPORTANT: Adjust sys.path to find your 'light_rag' package and venv site-packages ---
# This path should point to the directory *containing* the 'light_rag' folder.
# e.g., /home/shtlp_0170/Videos/i2/US-Tax-Law-RAG-Demo/
sys.path.append("/home/shtlpmac054/Documents/us_tax_law_rag/US_Tax_Law_RAG_Demo/")
# For the lightrag library itself within your virtual environment
sys.path.append("/home/shtlpmac054/Documents/us_tax_law_rag/US_Tax_Law_RAG_Demo/light_rag_env/lib/python3.10/site-packages")
 
 
# Load environment variables
load_dotenv()
 
# --- Environment Variables ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 1536)) # Ensure this is an integer
 
# --- Import LightRAG and your custom modules from the 'light_rag' package ---
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm import (
    azure_openai_complete,
    azure_openai_embedding
)
from pdf_loader import PDFLoader
from knowledge_base import KnowledgeBase
from _astradb import AstraDBKVStorage, AstraDBVectorStorage
 
 
# Initialize FastAPI app
app = FastAPI()
 
# --- Global instances for LightRAG and Knowledge Base ---
# These will be initialized once when the server starts.
rag_instance = None
knowledge_base_instance = None
 
# Pydantic model to define the structure of the incoming POST request body
class ChatRequest(BaseModel):
    user_query: str
    response_mode: str = "default" # Default value if not provided
    history: list[dict] = [] # List of {"role": str, "content": str}
 
 
# --- Prompt Templates (moved directly into the server logic) ---
# It's better to define these globally or load from config for a server
SUMMARY_PROMPT_TEMPLATE = """
You are a helpful assistant whose job is to respond to summarise case.
if the user ask to summarise the case use the below prompt format to generate case summary
 
### Prompt for Summary
 
Prompt:
 
Write a concise case summary for a legal case. The summary should include the following elements:
1.  Title and Citation : Provide the full case name and citation.
2.  Facts : Summarize the key facts and context of the case.
3.  Issues : Outline the main legal issues being addressed.
4.  Decision/Outcome : Describe the court's decision and its implications.
5.  Key Quotes or Principles : Include any significant quotes or legal principles from the judgment that highlight the case's importance.
 
Aim for a length of about 300-500 words, focusing on clarity and key points relevant to understanding the case.
Conversation History: {history}
User's Question: {user_query}
"""
 
DETAIL_CASE_PROMPT = """
If the user asks to give the detailing of the case use the prompt for detailing.
### Prompt for Detailed Case Brief
 
Prompt:
 
Create a detailed case brief for a legal case, covering the following components:
1.  Case Title and Citation : Include the full title and citation of the case.
2.  Parties Involved : Identify the appellant and respondent.
3.  Facts of the Case : Provide an in-depth description of the case background, including relevant events leading up to the litigation.
4.  Issues Before the Court : List the specific legal issues or questions that the court needed to resolve.
5.  Arguments : Summarize the key arguments presented by both the appellant and respondent.
6.  Judgment : Detail the court's decision and reasoning, including any significant legal tests or principles established.
7.  Conclusion : Discuss the implications of the judgment for future cases or legal precedents.
 
The brief should be comprehensive, approximately 700-1,000 words, and organized clearly for easy understanding.
Conversation History: {history}
User's Question: {user_query}
"""
 
PRESS_RELEASE_PROMPT = """
If the user asks for press release of the case use the prompt for press release.
### Prompt for Press Release Style Summary
 
Prompt:
 
Draft a press release summarizing a legal case decision. The press release should include:
1.  Title and Citation : State the full name and citation of the case.
2.  Date of Decision : Include the date the judgment was delivered.
3.  Court Justices : List the justices involved in the case.
4.  Background to the Appeal : Provide a brief overview of the case background and the legal issues raised.
5.  Judgment Summary : State whether the appeal was allowed or dismissed and summarize the court's reasoning.
6.  Key Reasons for Judgment : Highlight the major points of law established in the judgment and their significance.
7.  Conclusion : Reflect on the broader implications of the ruling for the legal landscape or affected parties.
8.  Disclaimer : Add a note that the summary is for informational purposes and does not substitute for the full judgment.
 
Ensure the tone is informative and accessible, targeting a general audience, and aim for a length of about 400-600 words.
Conversation History: {history}
User's Question: {user_query}
"""
 
 
@app.on_event("startup")
async def startup_event():
    """
    Initializes LightRAG and loads the knowledge base when the FastAPI application starts.
    This replaces the __init__ and knowledge base loading logic from your Streamlit app.
    """
    global rag_instance, knowledge_base_instance
    print("Initializing LightRAG and Knowledge Base for FastAPI service...")
 
    # Define working directory relative to this script's location
    working_dir = os.path.join(os.path.dirname(__file__), "chats_db")
    if not os.path.exists(working_dir):
        os.makedirs(working_dir) # Use makedirs to create intermediate dirs if needed
 
    vector_db_storage_cls_kwargs = {
        "ASTRADB_APPLICATION_TOKEN": os.getenv("ASTRADB_APPLICATION_TOKEN"),
        "ASTRADB_API_ENDPOINT": os.getenv("ASTRADB_API_ENDPOINT"),
        "EMBEDDING_DIMENSIONS": EMBEDDING_DIMENSIONS,
    }
 
    rag_instance = LightRAG(
        working_dir=working_dir,
        llm_model_func=azure_openai_complete,
        embedding_func=azure_openai_embedding,
        vector_storage="AstraDBVectorStorage",
        vector_db_storage_cls_kwargs=vector_db_storage_cls_kwargs,
    )
    # Correctly initialize text_chunks as per your original code
    rag_instance.text_chunks = AstraDBKVStorage(
        namespace="chunks",
        global_config=asdict(rag_instance),
        embedding_func=rag_instance.embedding_func
    )
 
    # Load your PDF content
    # Adjust this path to where your PDFs are located on the server
    # It seems like '/home/shtlp_0170/Videos/i2' is the directory containing your PDFs.
    pdf_loader = PDFLoader("../data")
    pdf_content = pdf_loader.load_contents()
 
    knowledge_base_instance = KnowledgeBase(rag_instance, pdf_content)
    await knowledge_base_instance.load() # Assuming knowledge_base.load() is asynchronous
 
    print("LightRAG and Knowledge Base initialized successfully for FastAPI service.")
 
 
@app.post("/chat_query")
async def chat_query_endpoint(request_data: ChatRequest):
    """
    Receives a POST request with user query, response mode, and history.
    Processes it using LightRAG and returns the response.
    """
    try:
        user_input = request_data.user_query
        response_mode = request_data.response_mode
        history_messages = request_data.history
 
        # Reconstruct history string from the list of messages
        history = ""
        # Use a consistent number of past messages, e.g., last 6 as in your Streamlit app
        previous_messages_for_prompt = history_messages[-6:]
        for message in previous_messages_for_prompt:
            role = "User" if message["role"] == "user" else "Assistant"
            history += f"{role}: {message['content']}\n"
 
        # Construct full_prompt based on response_mode
        full_prompt = ""
        if response_mode == "summary":
            full_prompt = SUMMARY_PROMPT_TEMPLATE.format(history=history.strip(), user_query=user_input)
        elif response_mode == "detailed_case":
            full_prompt = DETAIL_CASE_PROMPT.format(history=history.strip(), user_query=user_input)
        elif response_mode == "press_release":
            full_prompt = PRESS_RELEASE_PROMPT.format(history=history.strip(), user_query=user_input)
        else:
            # Default mode, similar to your original 'else'
            full_prompt = f"{history}**User:** {user_input}\n**Assistant:**"
 
        if rag_instance is None:
            raise RuntimeError("LightRAG instance not initialized.")
 
        # Perform the LightRAG query asynchronously
        response_text = await rag_instance.query(full_prompt, param=QueryParam(mode="hybrid"))
 
        return {"response": response_text}
 
    except Exception as e:
        print(f"Error processing chat query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during query processing: {str(e)}")
 
 
# To run this FastAPI service:
# 1. Save this code as 'lightrag_api_service.py' in your project root.
# 2. Make sure your virtual environment is activated.
# 3. Run from your terminal in the project root:
#    uvicorn lightrag_api_service:app --reload --port 8000
#
# You can then send POST requests to http://127.0.0.1:8000/chat_query
# with a JSON body like:
# {
#   "user_query": "Summarize the landmark case of Roe v. Wade.",
#   "response_mode": "summary",
#   "history": [
#     {"role": "user", "content": "Tell me about legal cases."},
#     {"role": "assistant", "content": "Which case would you like to know about?"}
#   ]
# }
