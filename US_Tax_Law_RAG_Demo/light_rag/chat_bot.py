import os
import streamlit as st
import time
from dataclasses import asdict



#added to solve error
import sys
sys.path.append("/Users/shtlpmac054/Documents/us_tax_law_rag/light_rag_env/lib/python3.10/site-packages")

# from lightrag import LightRAG, QueryParam  change to below line
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm import (
    gpt_4o_mini_complete,
    azure_openai_complete,
    azure_openai_embedding
)
# from lightrag.kg.astra_impl import AstraDBKVStorage
from pdf_loader import PDFLoader
from knowledge_base import KnowledgeBase
from _astradb import AstraDBKVStorage, AstraDBVectorStorage
# from _neo4j import Neo4JStorage
import numpy as np
import aiohttp
from dotenv import load_dotenv
from lightrag.utils import wrap_embedding_func_with_attrs

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import (
    APIConnectionError,
    RateLimitError,
    Timeout,
)

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
# )
# async def azure_openai_llm(
#     prompt, system_prompt=None, history_messages=[], **kwargs
# ) -> str:
#     headers = {
#         "Content-Type": "application/json",
#         "api-key": AZURE_OPENAI_API_KEY,
#     }
#     endpoint = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
#     print(endpoint)
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     if history_messages:
#         messages.extend(history_messages)
#     messages.append({"role": "user", "content": prompt})

#     payload = {
#         "messages": messages,
#         "temperature": kwargs.get("temperature", 0),
#         "top_p": kwargs.get("top_p", 1),
#         "n": kwargs.get("n", 1),
#     }

#     async with aiohttp.ClientSession() as session:
#         async with session.post(endpoint, headers=headers, json=payload) as response:
#             if response.status != 200:
#                 raise ValueError(
#                     f"Request failed with status {response.status}: {await response.text()}"
#                 )
#             result = await response.json()
#             return result["choices"][0]["message"]["content"]


# @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
# )
# async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
#     headers = {
#         "Content-Type": "application/json",
#         "api-key": AZURE_OPENAI_API_KEY,
#     }
#     endpoint = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_EMBEDDING_DEPLOYMENT}/embeddings?api-version={AZURE_EMBEDDING_API_VERSION}"
#     print(endpoint)
#     payload = {"input": texts}

#     async with aiohttp.ClientSession() as session:
#         async with session.post(endpoint, headers=headers, json=payload) as response:
#             if response.status != 200:
#                 raise ValueError(
#                     f"Request failed with status {response.status}: {await response.text()}"
#                 )
#             result = await response.json()
#             embeddings = [item["embedding"] for item in result["data"]]
#             return np.array(embeddings)


class ChatbotApp:
    def __init__(self):
        """Initialize the ChatbotApp with the required setup and knowledge base."""
        self.working_dir = "../chats_db"
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

        st.title("LightRAG Chatbot")
        # For AstraDB
        vector_db_storage_cls_kwargs = {
            "ASTRADB_APPLICATION_TOKEN": os.getenv("ASTRADB_APPLICATION_TOKEN"),
            "ASTRADB_API_ENDPOINT": os.getenv("ASTRADB_API_ENDPOINT"),
            "EMBEDDING_DIMENSIONS": os.getenv("EMBEDDING_DIMENSIONS"),
        }

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=azure_openai_complete,
            embedding_func=azure_openai_embedding,
            vector_storage="AstraDBVectorStorage",
            vector_db_storage_cls_kwargs=vector_db_storage_cls_kwargs,
            # graph_storage_cls=Neo4JStorage,
            # chunk_token_size=250,
            # chunk_overlap_token_size=25
        )

        self.rag.text_chunks = AstraDBKVStorage(namespace="chunks", global_config=asdict(self.rag), embedding_func=self.rag.embedding_func)
   
        if "knowledge_base_loaded" not in st.session_state:
            st.session_state.knowledge_base_loaded = False

        if not st.session_state.knowledge_base_loaded:
            with st.spinner("Loading knowledge base, please wait..."):
                time.sleep(1) 
                pdf_loader = PDFLoader("../data")
                pdf_content = pdf_loader.load_contents()
                self.knowledge_base = KnowledgeBase(self.rag, pdf_content)
                self.knowledge_base.load()

            st.session_state.knowledge_base_loaded = True  

        st.write("Knowledge base is ready! You can start chatting below.")
        self._initialize_ui()

    def _initialize_ui(self):
        """Set up the UI and session state for the chatbot app."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "response_mode" not in st.session_state:
            st.session_state.response_mode = "default"
        st.session_state.response_mode = st.selectbox(
            "Select Response Mode",
            options=["default", "summary", "detailed_case", "press_release"],
            index=["default", "summary", "detailed_case", "press_release"].index(st.session_state.response_mode)
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def get_user_input(self):
        """Capture and return user input from the chat input box."""
        return st.chat_input("Enter your question here:")

    def generate_response(self, user_input):
        """Generate a response from the model based on user input and update session state."""
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})

        history = ""
        previous_messages = st.session_state.messages[-6:]
        for message in previous_messages:
            role = "User" if message["role"] == "user" else "Assistant"
            history += f"{role}: {message['content']}\n"

        summary_prompt_template = """
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

        detail_case_prompt = """
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

        press_release_prompt = """
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

        if st.session_state.response_mode == "summary":
            full_prompt = summary_prompt_template.format(history=history.strip(), user_query=user_input)
        elif st.session_state.response_mode == "detailed_case":
            full_prompt = detail_case_prompt.format(history=history.strip(), user_query=user_input)
        elif st.session_state.response_mode == "press_release":
            full_prompt = press_release_prompt.format(history=history.strip(), user_query=user_input)
        else:
            full_prompt = f"{history}**User:** {user_input}\n**Assistant:**"

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = self.rag.query(full_prompt, param=QueryParam(mode="hybrid"))
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    def run(self):
        """Run the main loop to accept user input and generate responses continuously."""
        user_input = self.get_user_input()
        if user_input:
            self.generate_response(user_input)
