from lightrag import LightRAG
import asyncio
class KnowledgeBase:
    def __init__(self, rag: LightRAG, pdf_content: list):
        self.rag = rag
        self.pdf_content = pdf_content

    async def load(self):
        """Load knowledge base into LightRAG from the PDF content."""
        if self.pdf_content:
            await self.rag.insert(self.pdf_content)
            print("Knowledge base loaded successfully!")
        else:
            raise ValueError("Knowledge base not found!")
