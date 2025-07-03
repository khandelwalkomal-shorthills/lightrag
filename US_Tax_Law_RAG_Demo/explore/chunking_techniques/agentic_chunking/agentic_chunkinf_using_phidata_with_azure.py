# """
# pip install phidata 

# """

# from phi.agent import Agent
# from phi.document.chunking.agentic import AgenticChunking
# from phi.knowledge.pdf import PDFUrlKnowledgeBase
# from phi.vectordb.pgvector import PgVector
# # from langchain.chat_models.azure_openai import AzureChatOpenAI
# # from langchain_openai import AzureChatOpenAI
# # from langchain.chat_models.azure_openai import AzureChatOpenAI
# from phi.model.azure.openai_chat import AzureOpenAIChat


# from typing import Optional, Dict, Any, List

# from pydantic import BaseModel, ConfigDict

# from phi.embedder import Embedder


# # import openai
# # from langchain_openai import AzureOpenAI

# class Document(BaseModel):
#     """Model for managing a document"""

#     content: str
#     id: Optional[str] = None
#     name: Optional[str] = None
#     meta_data: Dict[str, Any] = {}
#     embedder: Optional[Embedder] = None
#     embedding: Optional[List[float]] = None
#     usage: Optional[Dict[str, Any]] = None
#     reranking_score: Optional[float] = None

#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     def embed(self, embedder: Optional[Embedder] = None) -> None:
#         """Embed the document using the provided embedder"""

#         _embedder = embedder or self.embedder
#         if _embedder is None:
#             raise ValueError("No embedder provided")

#         self.embedding, self.usage = _embedder.get_embedding_and_usage(self.content)

#     def to_dict(self) -> Dict[str, Any]:
#         """Returns a dictionary representation of the document"""

#         return self.model_dump(include={"name", "meta_data", "content"}, exclude_none=True)

#     @classmethod
#     def from_dict(cls, document: Dict[str, Any]) -> "Document":
#         """Returns a Document object from a dictionary representation"""

#         return cls.model_validate(**document)

#     @classmethod
#     def from_json(cls, document: str) -> "Document":
#         """Returns a Document object from a json string representation"""

#         return cls.model_validate_json(document)







# text_to_process = """
#     Artificial intelligence (AI) is transforming industries by enabling machines to perform tasks that
#     previously required human intelligence. From healthcare to finance, AI is driving innovation and improving
#     efficiency. For instance, in healthcare, AI algorithms assist doctors in diagnosing diseases, interpreting
#     medical images, and predicting patient outcomes. Meanwhile, in finance, AI helps detect fraud, manage
#     investments, and automate customer service.

#     However, the widespread adoption of AI also raises ethical concerns. Issues like privacy invasion,
#     algorithmic bias, and the potential loss of jobs due to automation are significant challenges. Experts
#     argue that it's essential to develop AI responsibly to ensure that it benefits society as a whole.
#     Proper regulations, transparency, and accountability can help address these issues, ensuring that AI
#     technologies are used for the greater good.

#     Beyond individual industries, AI is also impacting the global economy. Nations are investing heavily
#     in AI research and development to maintain a competitive edge. This technological race could redefine
#     global power dynamics, with countries that excel in AI leading the way in economic and military strength.
#     Despite the potential for AI to contribute positively to society, its development and application require
#     careful consideration of ethical, legal, and societal implications.
#     """



# from dotenv import load_dotenv
# import openai
# import os
# import langchain

# load_dotenv()
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
# AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")
# OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")

# openai.api_key = OPENAI_API_KEY
# # openai.azure_endpoint = AZURE_OPENAI_ENDPOINT

# # Pull the prebuilt proposition extraction prompt from LangHub

# llm = AzureOpenAIChat(
#     id = "model-1",
#     api_key=AZURE_OPENAI_API_KEY,
#     azure_deployment=AZURE_OPENAI_DEPLOYMENT,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT
# )

# # doc = Document(content =text_to_process)
# # ag = AgenticChunking(model=llm, max_chunk_size=500)
# # print("_"*100)
# # print("model_data", ag.model)
# # print(ag.chunk(doc))
# # print("_"*100)
# # exit()


# # Initialize GPT-4 with your Azure deployment info
# # llm = AzureChatOpenAI(
# #     azure_deployment=AZURE_OPENAI_DEPLOYMENT,
# #     model='gpt-4-1106-preview',
# #     api_version=AZURE_OPENAI_API_VERSION,
# #     api_key=AZURE_OPENAI_API_KEY
# # )
# # print(llm.invoke("How are your"))

# # llm = AzureOpenAIChat(id = "1234", api_version=AZURE_OPENAI_API_VERSION)
# # print(llm.invoke("How are you"))
# # (
  


# #     # azure_deployment=AZURE_OPENAI_DEPLOYMENT,
# #     # model='gpt-4-1106-preview',
# #     # api_version=AZURE_OPENAI_API_VERSION
# # )


# # db_url = "postgresql://@localhost:5532/ai"
# # db_url = "postgresql://shtlpmac_049:postgres@localhost:5432/ai"
# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"




# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=PgVector(table_name="recipes_agentic_chunking", db_url=db_url),
#     chunking_strategy=AgenticChunking(max_chunk_size=500),
# )
# knowledge_base.load(recreate=False)  # Comment out after first run

# agent = Agent(
#     knowledge_base=knowledge_base,
#     search_knowledge=True,
# )

# agent.print_response("How to make Thai curry?", markdown=True)



from phi.agent import Agent
from phi.document.chunking.agentic import AgenticChunking
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector
import os
import openai
from dotenv import load_dotenv
load_dotenv()
from phi.model.azure import AzureOpenAIChat




azure_model = AzureOpenAIChat(
    id=os.getenv("AZURE_OPENAI_MODEL_NAME") or "gpt-4",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)


# OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")
# openai.api_key = OPENAI_API_KEY

# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db_url = "postgresql://shtlpmac_049:postgres@localhost:5432/ai"

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes_agentic_chunking", db_url=db_url),
    chunking_strategy=AgenticChunking(model=azure_model, max_chunk_size=1200),
)
knowledge_base.load(recreate=False)  # Comment out after first run

agent = Agent(
    knowledge_base=knowledge_base,
    search_knowledge=True,
)

agent.print_response("give me some thai food name", markdown=True)


