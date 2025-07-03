from typing import List, Optional
from phi.model.azure import AzureOpenAIChat
from phi.model.openai import OpenAIChat
import os
# Placeholder classes to simulate the environment
# You would replace these with your actual imports like phi.document.base.Document, etc.

azure_model = AzureOpenAIChat(
    id=os.getenv("AZURE_OPENAI_MODEL_NAME") or "gpt-4",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

class Document:
    def __init__(self, id: str, name: str, meta_data: dict, content: str):
        self.id = id
        self.name = name
        self.meta_data = meta_data
        self.content = content

    def __repr__(self):
        return f"Document(id={self.id}, name={self.name}, meta_data={self.meta_data}, content={self.content}...)"

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# AgenticChunking Class
class AgenticChunking:
    """
    Chunking strategy that uses an LLM to determine natural breakpoints in the text.
    """

    def __init__(self, model: Optional[OpenAIChat] = None, max_chunk_size: int = 1200):
        self.model = model or OpenAIChat()
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> List[Document]:
        """
        Chunk a document into smaller, semantically meaningful pieces using an LLM.
        
        Args:
            document (Document): The document to be chunked.
        
        Returns:
            List[Document]: A list of smaller documents (chunks).
        """
        # If the document is smaller than the max chunk size, return as a single chunk
        if len(document.content) <= self.max_chunk_size:
            return [document]

        # Initialize variables
        chunks: List[Document] = []  # List to store chunked documents
        remaining_text = self.clean_text(document.content)  # Cleaned text for processing
        chunk_meta_data = document.meta_data  # Metadata to be carried over to each chunk
        chunk_number = 1  # Start chunk numbering from 1

        # Continue chunking until all the text is processed
        while remaining_text:
            # Create a prompt asking the model to determine a natural breakpoint
            prompt = f"""
            Analyze this text and determine a natural breakpoint within the first {self.max_chunk_size} characters. 
            Consider semantic completeness, paragraph boundaries, and topic transitions.
            Return only the character position number where to break the text:

            {remaining_text[:self.max_chunk_size]}
            """

            try:
                # Get model's response to the prompt
                response = self.model.response([Message(role="user", content=prompt)])
                if response and response.content:
                    # Extract the breakpoint from the model's response
                    break_point = min(int(response.content.strip()), self.max_chunk_size)
                else:
                    # Default to max_chunk_size if no valid response
                    break_point = self.max_chunk_size
            except Exception:
                # In case of an error, fall back to the max_chunk_size
                break_point = self.max_chunk_size

            # Extract the chunk of text up to the determined breakpoint
            chunk = remaining_text[:break_point].strip()
            # Prepare metadata for this chunk
            meta_data = chunk_meta_data.copy()
            meta_data["chunk"] = chunk_number  # Add chunk number to metadata
            chunk_id = f"{document.id}_{chunk_number}" if document.id else f"{document.name}_{chunk_number}"
            meta_data["chunk_size"] = len(chunk)  # Record the chunk size in metadata

            # Append the chunk as a new Document object to the chunks list
            chunks.append(
                Document(
                    id=chunk_id,
                    name=document.name,
                    meta_data=meta_data,
                    content=chunk,
                )
            )

            # Update chunk number for the next iteration
            chunk_number += 1
            # Update the remaining text to the part after the current chunk
            remaining_text = remaining_text[break_point:].strip()

            # If there's no more text left, break out of the loop
            if not remaining_text:
                break

        return chunks

    def clean_text(self, text: str) -> str:
        """Clean the text by replacing multiple newlines with a single newline"""
        import re

        # Replace multiple newlines with a single newline
        cleaned_text = re.sub(r"\n+", "\n", text)
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        # Replace multiple tabs with a single tab
        cleaned_text = re.sub(r"\t+", "\t", cleaned_text)
        # Replace multiple carriage returns with a single carriage return
        cleaned_text = re.sub(r"\r+", "\r", cleaned_text)
        # Replace multiple form feeds with a single form feed
        cleaned_text = re.sub(r"\f+", "\f", cleaned_text)
        # Replace multiple vertical tabs with a single vertical tab
        cleaned_text = re.sub(r"\v+", "\v", cleaned_text)

        return cleaned_text


# Simulate a sample document and use the AgenticChunking strategy
sample_text = """
This is a long document that we want to chunk into multiple pieces. 
We need to ensure that the chunking is done based on natural breakpoints in the text.
This includes considering paragraph boundaries, topic transitions, and semantic completeness.
The model should help in determining the optimal way to break the text without losing important context.
Here is some more content to simulate a longer document. 
The chunking should be intelligent and adapt to the content, allowing us to process it more effectively.
"""

# Create a sample document
doc = Document(id="doc_1", name="Sample Document", meta_data={"author": "GPT", "type": "text"}, content=sample_text)

# Initialize the chunking strategy
chunking_strategy = AgenticChunking(model=azure_model, max_chunk_size=500)

# Perform the chunking operation
chunks = chunking_strategy.chunk(doc)

# Display the chunked documents
for chunk in chunks:
    print(f"Chunk ID: {chunk.id}, Chunk Size: {len(chunk.content)}")
    print(f"Content:{chunk}")  # Display first 100 characters
