from abc import ABC, abstractmethod
from typing import List, Dict
from typing import Optional, Callable
from lightrag import LightRAG
from lightrag.lightrag import always_get_an_event_loop
from lightrag.operate import extract_entities
from dataclasses import asdict, dataclass, field
from lightrag.utils import decode_tokens_by_tiktoken,encode_string_by_tiktoken,compute_mdhash_id
import logging
import os
from lightrag.llm import (
    gpt_4o_mini_complete,
)

working_dir = "../llm_chats_db"  # Correct: variable assignment, no quotes around variable name
if not os.path.exists(working_dir):
    os.mkdir(working_dir)


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]); logger = logging.getLogger('my_logger')


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    """

    @abstractmethod
    def chunk(self, content: str, overlap_token_size: int, max_token_size: int, tiktoken_model: str) -> List[Dict]:
        """
        Method to chunk content based on the specific strategy.
        
        :param content: The content to be chunked.
        :param overlap_token_size: The number of overlapping tokens.
        :param max_token_size: The maximum token size for each chunk.
        :param tiktoken_model: The model to use for tokenizing the content.
        :return: A list of chunks with relevant information.
        """
        pass

class ChunkingByTokenSize(ChunkingStrategy):
    """
    Chunking strategy based on token size using tiktoken.
    """

    def chunk(self, content: str, overlap_token_size: int, max_token_size: int, tiktoken_model: str) -> List[Dict]:
        tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
        chunks = []
        for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
            chunk_content = decode_tokens_by_tiktoken(tokens[start:start + max_token_size], model_name=tiktoken_model)
            chunks.append({
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            })
        return chunks

class ChunkingByFixedSize(ChunkingStrategy):
    """
    Chunking strategy based on fixed content size.
    """

    def chunk(self, content: str, overlap_token_size: int, max_token_size: int, tiktoken_model: str) -> List[Dict]:
        chunks = []
        for i in range(0, len(content), max_token_size):
            chunk_content = content[i:i + max_token_size]
            chunks.append({
                "tokens": len(chunk_content.split()),  # Approximate token count by splitting
                "content": chunk_content.strip(),
                "chunk_order_index": len(chunks),
            })
        return chunks



class ModularLightRAG(LightRAG):
    """
    A subclass of LightRAG that supports dynamic chunking strategies.
    """

    def __init__(self, *args, **kwargs):
        # Initialize LightRAG with default parameters
        super().__init__(*args, **kwargs)
    
    def insert(self, string_or_strings, chunking_func: Optional[ChunkingStrategy] = None):
        """
        Insert documents into storage with a dynamic chunking strategy.
        
        :param string_or_strings: The document(s) to insert.
        :param chunking_func: The chunking function to use. Defaults to `ChunkingByTokenSize` if not provided.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings, chunking_func))

    async def ainsert(self, string_or_strings, chunking_func: Optional[ChunkingStrategy] = None):
        """
        Async insert method with support for different chunking strategies.
        """
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                self.compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # Select chunking function (use passed one or default to ChunkingByTokenSize)
            chunking_func = chunking_func or ChunkingByTokenSize()

            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_func.chunk(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            # Entity extraction and relationship handling
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            # Upsert full docs and text chunks
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            if update_storage:
                await self._insert_done()



class ChatbotApp:
    def __init__(self):
        """Initialize the ChatbotApp with the required setup and knowledge base."""
        # self.working_dir = "../chats_db"
        self.working_dir = "../llm_chats_db"
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

        self.rag = LightRAG(working_dir=self.working_dir, llm_model_func=gpt_4o_mini_complete)



