import langchain.chains
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType
from dotenv import load_dotenv
import openai
import os
import langchain

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


# Initialize GPT-4 with your Azure deployment info
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    model='gpt-4-1106-preview',
    api_version=AZURE_OPENAI_API_VERSION
)


# Initialize OpenAI chat model (replace with your API key)
# llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="replace with your actual OpenAI API key")

# Step 1: Define Chunking and Summarization Prompt Template
chunk_prompt_template = """
You are given a large piece of text. Your job is to break it into smaller parts (chunks) if necessary and summarize each chunk.
Once all parts are summarized, combine them into a final summary. 
If the text is already small enough to process at once, provide a full summary in one step. 
Please summarize the following text:\n{input}
"""
chunk_prompt = PromptTemplate(input_variables=["input"], template=chunk_prompt_template)

# Step 2: Define Chunk Processing Tool
def chunk_processing_tool(query):
    """Processes text chunks and generates summaries using the defined prompt."""
    chunk_chain = LLMChain(llm=llm, prompt=chunk_prompt)
    
    print(f"Processing chunk:\n{query}\n")  # Show the chunk being processed
    return chunk_chain.run(input=query)
    

# Step 3: Define External Tool (Optional, can be used to fetch extra information if needed)
def external_tool(query):
    """Simulates an external tool that could fetch additional information."""
    return f"External response based on the query: {query}"

# Step 4: Initialize the agent with tools
tools = [
    Tool(
        name="Chunk Processing",
        func=chunk_processing_tool,
        description="Processes text chunks and generates summaries."
    ),
    Tool(
        name="External Query",
        func=external_tool,
        description="Fetches additional data to enhance chunk processing."
    )
]

# Initialize the agent with defined tools and zero-shot capabilities
agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)

# Step 5: Agentic Chunk Processing Function
def agent_process_chunks(text):
    """Uses the agent to process text chunks and generate a final output."""
    # Step 1: Chunking the text into smaller, manageable sections
    def chunk_text(text, chunk_size=1200):
        """Splits large text into smaller chunks."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    chunks = chunk_text(text)
    
    # Step 2: Process each chunk with the agent
    chunk_results = []
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx + 1}/{len(chunks)}...")
        response = agent.invoke({"input": chunk})  # Process chunk using the agent
        chunk_results.append(response['output'])  # Collect the chunk result

    # Step 3: Combine the chunk results into a final output
    # final_output = "\n".join(chunk_results)
    return chunk_results
    return final_output

# Step 6: Running the agent on an example large text input
if __name__ == "__main__":
    # Example large text content
    text_to_process = """
    Artificial intelligence (AI) is transforming industries by enabling machines to perform tasks that
    previously required human intelligence. From healthcare to finance, AI is driving innovation and improving
    efficiency. For instance, in healthcare, AI algorithms assist doctors in diagnosing diseases, interpreting
    medical images, and predicting patient outcomes. Meanwhile, in finance, AI helps detect fraud, manage
    investments, and automate customer service.

    However, the widespread adoption of AI also raises ethical concerns. Issues like privacy invasion,
    algorithmic bias, and the potential loss of jobs due to automation are significant challenges. Experts
    argue that it's essential to develop AI responsibly to ensure that it benefits society as a whole.
    Proper regulations, transparency, and accountability can help address these issues, ensuring that AI
    technologies are used for the greater good.

    Beyond individual industries, AI is also impacting the global economy. Nations are investing heavily
    in AI research and development to maintain a competitive edge. This technological race could redefine
    global power dynamics, with countries that excel in AI leading the way in economic and military strength.
    Despite the potential for AI to contribute positively to society, its development and application require
    careful consideration of ethical, legal, and societal implications.
    """

    # Process the text and print the final result
    final_result = agent_process_chunks(text_to_process)
    print("\nFinal Output:\n", final_result)