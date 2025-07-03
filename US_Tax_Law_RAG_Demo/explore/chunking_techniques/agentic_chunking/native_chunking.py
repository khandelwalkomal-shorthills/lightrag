from langchain import hub
# from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain.chat_models.azure_openai import AzureChatOpenAI


from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub

import os
from dotenv import load_dotenv
import openai
from pydantic import BaseModel, Field  # Pydantic V2 import
from typing import List
from langchain.chains.openai_functions import create_extraction_chain_pydantic  # Correct import
import PyPDF2
import json  # For parsing JSON strings

# Load environment variables from the .env file
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

openai.api_key = AZURE_OPENAI_API_KEY
openai.azure_endpoint = AZURE_OPENAI_ENDPOINT

# Pull the prebuilt proposition extraction prompt from LangHub
obj = hub.pull("wfh/proposal-indexing")

# Initialize GPT-4 with your Azure deployment info
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    model='gpt-4o-mini',
    api_version=AZURE_OPENAI_API_VERSION
)

# llm = AzureChatOpenAI(
#     azure_deployment=AZURE_OPENAI_DEPLOYMENT,
#     model='gpt-4-1106-preview',
#     api_version=AZURE_OPENAI_API_VERSION
# )


# Combine the prompt and the language model into a single runnable
runnable = (obj | llm)

# Define the Pydantic V2 model for sentence extraction
class Sentences(BaseModel):
    sentences: List[str]  # Pydantic V2 models use `Field` to specify attributes

# Create the extraction chain using the updated model
extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)

def get_propositions(text):
    # Run the text through the runnable
    result = runnable.invoke({"input": text})

    # Access the content from the result
    content = result.content

    # Parse the JSON string in the content field (check for format)
    try:
        # Clean up any extra markdown formatting and parse the content
        content = content.strip("```json\n").strip()  # Remove any markdown formatting
        parsed_content = json.loads(content)  # Attempt to parse as JSON
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        parsed_content = []  # Default to an empty list if parsing fails

    # Check the structure of parsed_content
    print(f"Parsed content: {parsed_content}")

    # Assuming parsed_content is a list of sentences
    propositions = parsed_content if isinstance(parsed_content, list) else []
    
    # Use extraction_chain to further process the sentences (if needed)
    if propositions:
        # If propositions are found, run the extraction chain
        extraction_result = extraction_chain.invoke(" ".join(propositions))  # Invoke the extraction chain
        print("Extraction result:", extraction_result)
        
        # Check the structure of the extraction_result
        if isinstance(extraction_result, dict) and 'text' in extraction_result:
            # Since 'text' is a list of Sentences, access the first Sentences object
            sentences_obj = extraction_result['text'][0]  # Get the first Sentences object
            propositions = sentences_obj.sentences if isinstance(sentences_obj, Sentences) else []

    # Print the extracted propositions
    print("_" * 100)
    # print(propositions)
    print("_" * 100)

    return propositions

# Open the PDF file and extract text from all pages
with open('/Users/shtlpmac_049/Desktop/chunking-techniques/036ba62d-029a-4be0-9646-782b92756c33.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    
    essay = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        # Safely extract text, handling empty or invalid text gracefully
        essay += page.extract_text().strip() or ""  

# Split the essay into paragraphs (each paragraph is treated as an independent chunk)
paragraphs = essay.split("\n\n")

# Collect propositions from each paragraph
essay_propositions = []

for i, para in enumerate(paragraphs[:1]):  # Limit to the first 5 paragraphs for testing
    propositions = get_propositions(para)
    essay_propositions.extend(propositions)
    # print(f"Done with paragraph {i}")

# Now essay_propositions contains all the propositions
# print(f"Total Propositions: {len(essay_propositions)}")
# print(essay_propositions[:10])  # Print first 10 propositions


from agentic_chunker import AgenticChunker

# Initialize the chunker
ac = AgenticChunker()

# Pass the extracted propositions into the chunker
ac.add_propositions(essay_propositions)

ac.pretty_print_chunks()

