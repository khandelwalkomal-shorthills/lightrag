import os
import base64
import io
from dotenv import load_dotenv
from PIL import Image
import openai
from unstructured.partition.pdf import partition_pdf

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and endpoints from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

# Set up OpenAI API key and Azure endpoint for OpenAI
openai.api_key = AZURE_API_KEY
openai.api_base = AZURE_ENDPOINT
openai.api_type = "azure"
openai.api_version = AZURE_OPENAI_API_VERSION

MODEL = "gpt-4"

prompt = "Once upon a time, there was a small village surrounded by mountains. One day,"

filepath = "/Users/shtlpmac_049/Desktop/chunking-techniques/Screenshot_with_image.pdf"



# raw_pdf_elements = partition_pdf(
#     filename=filepath,
    
#     # Using pdf format to find embedded image blocks
#     extract_images_in_pdf=True,
    
#     # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
#     # Titles are any sub-section of the document
#     infer_table_structure=True,
    
#     # Post processing to aggregate text once we have the title
#     chunking_strategy="by_title",
#     # Chunking params to aggregate text blocks
#     # Attempt to create a new chunk 3800 chars
#     # Attempt to keep chunks > 2000 chars
#     # Hard max on chunks
#     max_characters=4000,
#     new_after_n_chars=3800,
#     combine_text_under_n_chars=2000,
#     image_output_dir_path="..static/pdfImages/",
# )
# exit()
# def image_to_base64(image_path: str) -> str:
#             with Image.open(image_path) as image:
#                 buffered = io.BytesIO()
#                 image.save(buffered, format=image.format)
#                 img_str = base64.b64encode(buffered.getvalue())
#                 return img_str.decode('utf-8')



def image_to_base64(image_path: str) -> str:
    with Image.open(image_path) as image:
        # Resize image to reduce size (example: reduce to 500px width)
        image.thumbnail((200, 200))
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode('utf-8')

        # Convert the image to base64
image_base64 = image_to_base64(image_path = "/Users/shtlpmac_049/Desktop/chunking-techniques/Screenshot_.png")
# exit()

messages = [
    {"role": "system", "content": "Please give a summary of the image provided. Be descriptive"},
    {"role": "user", "content": f"data:image/jpeg;base64,{image_base64}"}
]

SYS_PROMPT="You are an software developer and solution architect having descent reasearch background.Need one chunk for the input image. Consider all texts and all the screenshots/workflows/figures/graphs/captions embedded in the input image. The chunk is required to develop GenAI powered RAG. Don't start with 'the image of a..', return the chunk only"
messages=[
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": [
        {"type": "text", "text": "Describe the images, dont miss any important information?"},
        {"type": "image_url", "/Users/shtlpmac_049/Desktop/chunking-techniques/Screenshot_.png": {
            "url": f"data:image/png;base64,{image_base64}"}
        }
    ]}
],

sys_prompt = (
        "You are a software developer and solution architect with a research background. "
        "Provide a detailed description of the image. Consider all the text, screenshots, workflows, "
        "figures, graphs, and captions embedded in the image. The chunk is required for developing a "
        "GenAI-powered RAG. Do not start with 'the image of a...', return the chunk only."
    )


# endpoint = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
endpoint  = AZURE_OPENAI_ENDPOINT
response = openai.chat.completions.create(
            model=MODEL,  # Specify the model (e.g., 'gpt-4')
            messages=messages,
            temperature=0.7,  # Control randomness of output
            max_tokens=100,  # Limit the response size
        )
# print(response)
# exit()

# print(response)
# exit()



SYS_PROMPT="You are an software developer and solution architect having descent reasearch background.Need one chunk for the input image. Consider all texts and all the screenshots/workflows/figures/graphs/captions embedded in the input image. The chunk is required to develop GenAI powered RAG. Don't start with 'the image of a..', return the chunk only"
#Image to text chunk using GPT-4o
response = client.chat.completions.create(
model=AOAI_DEPLOYMENT,
messages=[
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": [
        {"type": "text", "text": "Describe the images, dont miss any important information?"},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{base64_image}"}
        }
    ]}
],
temperature=0.0,
)
chunk=response.choices[0].message.content