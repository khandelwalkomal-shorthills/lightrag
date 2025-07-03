import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker

# Load environment variables from a .env file (if needed)
load_dotenv()

# Helper function to read PDF content and return as a string
def read_pdf_to_string(file_path):
    """
    Reads a PDF file and extracts its text content.

    Args:
    - file_path (str): Path to the PDF file.

    Returns:
    - str: Extracted text from the PDF.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""  # Safely concatenate text from each page
    return text

# Define the file path to the PDF document
pdf_path = "/Users/shtlpmac_049/Desktop/chunking-techniques/036ba62d-029a-4be0-9646-782b92756c33.pdf"

# Read the content of the PDF into a string
content = read_pdf_to_string(pdf_path)

# Parameters for Semantic Chunking
breakpoint_type = 'percentile'  # Split text based on sentence difference percentile
threshold_value = 90  # The threshold value (percentile) to trigger chunking

# Initialize Azure OpenAI Embeddings (with a chunk size limit)
embeddings = AzureOpenAIEmbeddings(chunk_size=512)

# Initialize the SemanticChunker with the chosen settings
text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type=breakpoint_type,
    breakpoint_threshold_amount=threshold_value
)

# Split the document content into semantically meaningful chunks
docs = text_splitter.create_documents([content])

# Output the number of resulting chunks and preview the first 200 characters of each
print(f"Number of chunks: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"Chunk {i + 1}: {doc.page_content[:200]}...")  # Displaying the first 200 characters of each chunk

# Create a vector store using FAISS for efficient similarity search
vectorstore = FAISS.from_documents(docs, embeddings)

# Setup a retriever to fetch the top 2 most relevant chunks for a query
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Define a sample query related to climate change
test_query = "What is the main cause of climate change?"

# Retrieve the most relevant chunks for the query
context = retriever.get_relevant_documents(test_query)

# Output the relevant context for the given query (showing the first 200 characters of each chunk)
print(f"\nContext for query: '{test_query}'")
for i, chunk in enumerate(context):
    print(f"Chunk {i + 1}: {chunk.page_content[:200]}...")  # Displaying the first 200 characters of each chunk
