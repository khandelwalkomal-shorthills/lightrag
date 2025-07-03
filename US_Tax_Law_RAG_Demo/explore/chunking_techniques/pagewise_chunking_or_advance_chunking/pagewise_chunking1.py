# import os
# import base64
# import io
# from dotenv import load_dotenv
# from PIL import Image
# import openai
# from unstructured.partition.pdf import partition_pdf
# import pytesseract

# # Load environment variables from .env file
# load_dotenv()

# # Retrieve API keys and endpoints from environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
# AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

# # Set up OpenAI API key and Azure endpoint for OpenAI
# openai.api_key = AZURE_API_KEY
# openai.api_base = AZURE_ENDPOINT
# openai.api_type = "azure"
# openai.api_version = AZURE_OPENAI_API_VERSION

# MODEL = "gpt-4o-mini"


# def image_to_base64(image_path: str) -> str:
#     with Image.open(image_path) as image:
#         # Resize image to reduce size (example: reduce to 500px width)
#         image.thumbnail((200, 200))
#         buffered = io.BytesIO()
#         image.save(buffered, format=image.format)
#         img_str = base64.b64encode(buffered.getvalue())
#         return img_str.decode('utf-8')
    
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")
    

# def extract_text_from_image(image_path):
#     try:
#         image = Image.open(image_path)  # Open the image
#         text = pytesseract.image_to_string(image)  # Use OCR to extract text
#         print(f"Extracted Text: \n{text}")
#         return text
#     except Exception as e:
#         print(f"Error extracting text using OCR: {e}")
#         return ""    


# # image_base64 = image_to_base64(image_path = "/Users/shtlpmac_049/Desktop/chunking-techniques/Screenshot_.png")
# # image_base64 = image_to_base64(image_path = "/Users/shtlpmac_049/Desktop/chunking-techniques/docter_report.webp")
# image_base64 = encode_image(image_path = "/Users/shtlpmac_049/Desktop/chunking-techniques/docter_report.png")
# print(image_base64)
# exit()
# # text = extract_text_from_image(image_path = "/Users/shtlpmac_049/Desktop/chunking-techniques/image.png")
# # print(text)


# # print(image_base64)
# # exit()
# sys_prompt = (
#         "You are a software developer and solution architect with a research background. "
#         "Provide a detailed description of the image. Consider all the text, screenshots, workflows, "
#         "figures, graphs, and captions embedded in the image. The chunk is required for developing a "
#         "GenAI-powered RAG. Do not start with 'the image of a...', return the chunk only. return all content in  text"
#     )


# response = openai.chat.completions.create(
#             model=MODEL,
#             messages=[
#                 {"role": "system", "content": sys_prompt},
#                 {"role": "user", "content": [
#                     {"type": "text", "text": "Describe the image, don't miss any important information."},
#                     {"type": "image_url", "image_url": {
#                         "url": f"data:image/png;base64,{image_base64}"
#                     }}
#                 ]},
#             ],
            
#         )
# chunk = response.choices[0].message.content
# print(chunk)

# exit()
            


import os
from pypdf import PdfReader
import fitz  # pymupdf
from langchain.docstore.document import Document
import openai
import base64


AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

# Set up OpenAI API key and Azure endpoint for OpenAI
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_type = "azure"
openai.api_version = AZURE_OPENAI_API_VERSION

MODEL = "gpt-4o-mini"


# Function to extract text and chunk PDF page-by-page
def extract_and_chunk_pdf(path):
    # Create a PDF reader object
    reader_object = PdfReader(path)
    
    # Initialize an empty list to store chunks
    chunk_list_pagewise = []

    # Iterate through each page of the PDF
    for i in range(len(reader_object.pages)):
        page = reader_object.pages[i]
        text = page.extract_text()  # Extract text from the page
        
        # Create a Document chunk with the page content and metadata
        chunk = Document(page_content=text, metadata={"filename": path, "page_number": (i + 1)})
        chunk_list_pagewise.append(chunk)
    
    return chunk_list_pagewise

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
# print(extract_and_chunk_pdf("/Users/shtlpmac_049/Desktop/chunking-techniques/paper_pdf.pdf") )   
    
# exit()

# Function to identify image-heavy pages based on embedded images
def identify_image_heavy_pages(pdf_path, image_area_threshold=0.05):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    image_heavy_pages = []

    # Iterate through all pages
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)

        # Get all images on the page
        images = page.get_images(full=True)

        if images:
            total_image_area = 0

            # Iterate through all the images on the page
            for img in images:
                xref = img[0]  # Image reference number
                
                # Get the image rectangle (bounding box)
                img_rects = page.get_image_rects(xref)

                if img_rects:
                    for rect in img_rects:
                        # Calculate the area of the image
                        image_area = (rect.width) * (rect.height)
                        total_image_area += image_area
                else:
                    print(f"No rectangles found for image with xref: {xref}")

            # Get the total page area (width * height)
            page_width, page_height = page.rect.width, page.rect.height
            total_page_area = page_width * page_height
            print("total_area", total_page_area)

            # Calculate the proportion of the page that is image
            image_area_ratio = total_image_area / total_page_area
            print("\nImage_area_ratio",image_area_ratio)

            # Classify the page as "image-heavy" if the image area exceeds the threshold
            if image_area_ratio > image_area_threshold:
                image_heavy_pages.append(page_num)

    return image_heavy_pages

def get_gpt4_image_description(image_str):
    # Open the image file in binary format (GPT-4 or similar model needs image input)
    # with open(image_path, "rb") as img_file:
    #     image_data = img_file.read()

    sys_prompt = (
        "You are a software developer and solution architect with a research background. "
        "Provide a detailed description of the image. Consider all the text, screenshots, workflows, "
        "figures, graphs, and captions embedded in the image. The chunk is required for developing a "
        "GenAI-powered RAG. Do not start with 'the image of a...', return the chunk only. return all content in  text"
    )
    

    # Send the image to GPT-4 (or other image-processing LLMs)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe the image, don't miss any important information."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_str}"
                    }}
                ]},
            ],  # This could be specific depending on your use case
        # GPT-4 model or any model suitable for image analysis
    )
    
    # Extract textual description from GPT-4's response (assuming 'text' is returned in response)
    description = response.choices[0].message.content
    return description



# Function to convert image-heavy pages to PNG and use GPT-4 for analysis
def convert_images_and_get_gpt4_description(pdf_path, image_heavy_pages, image_folder="image_folder-1"):
    pdf_document = fitz.open(pdf_path)  # Open the PDF document
    image_descriptions = {}  # Dictionary to store descriptions for each image-heavy page

    # Ensure the image folder exists
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    for page_num in image_heavy_pages:
        page = pdf_document.load_page(page_num)  # Load the page
        pix = page.get_pixmap()  # Render the page as an image
        
        # Save the page as a PNG file
        image_path = os.path.join(image_folder, f"page-{page_num}.png")
        pix.save(image_path)
        image_str = encode_image(image_path)
        # Send the image to GPT-4 for description (You need to implement your GPT-4 call here)
        description = get_gpt4_image_description(image_str)
        
        # Store the description of the image
        image_descriptions[page_num] = description

    return image_descriptions
# print(convert_images_and_get_gpt4_description("/Users/shtlpmac_049/Desktop/chunking-techniques/paper_pdf.pdf", [0], ))
# exit()
# Function to generate description from GPT-4 (assuming GPT-4 integration is set up)

# Main function to process the PDF document
def process_pdf_for_rag(pdf_path):
    # Step 1: Extract text from the PDF and chunk it page-wise
    text_chunks = extract_and_chunk_pdf(pdf_path)
    print(text_chunks)
    # exit()

    # Step 2: Identify image-heavy pages
    image_heavy_pages = identify_image_heavy_pages(pdf_path)

    # Step 3: Convert image-heavy pages to images and get descriptions from GPT-4
    image_descriptions = convert_images_and_get_gpt4_description(pdf_path, image_heavy_pages)

    # Step 4: Replace regular text chunks with GPT-4 generated text for image-heavy pages
    updated_chunks = []
    for chunk in text_chunks:
        # If the chunk corresponds to an image-heavy page, replace it with the GPT-4 description
        if chunk.metadata["page_number"] in image_descriptions:
            description = image_descriptions[chunk.metadata["page_number"]]
            updated_chunk = Document(page_content=description, metadata=chunk.metadata)
            updated_chunks.append(updated_chunk)
        else:
            updated_chunks.append(chunk)
    
    return updated_chunks

if __name__ == "__main__":


    pdf_path = "/Users/shtlpmac_049/Desktop/chunking-techniques/paper_pdf.pdf"  # Set your PDF path here
    updated_chunks = process_pdf_for_rag(pdf_path)

    # Example: Print the updated chunks/Users/shtlpmac_049/Downloads/paper_pdf.pdf/Users/shtlpmac_049/Downloads/paper_pdf.pdf
    for chunk in updated_chunks:
        print("_"*100)
        print(f"Page {chunk.metadata['page_number']}:\n{chunk.page_content}\n")
        print("_"*100)


import os
import base64
from pypdf import PdfReader
import fitz  # pymupdf
from langchain.docstore.document import Document
import openai

# Set up environment variables and OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_type = "azure"
openai.api_version = AZURE_OPENAI_API_VERSION

MODEL = "gpt-4o-mini"

# Utility functions

def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# PDF Processing Functions

def extract_and_chunk_pdf(pdf_path):
    """Extracts text from a PDF and chunks it page-wise."""
    reader_object = PdfReader(pdf_path)
    chunk_list_pagewise = []

    for i in range(len(reader_object.pages)):
        page = reader_object.pages[i]
        text = page.extract_text()
        chunk = Document(page_content=text, metadata={"filename": pdf_path, "page_number": i + 1})
        chunk_list_pagewise.append(chunk)

    return chunk_list_pagewise

def identify_image_heavy_pages(pdf_path, image_area_threshold=0.05):
    """Identifies image-heavy pages in a PDF based on embedded images."""
    pdf_document = fitz.open(pdf_path)
    image_heavy_pages = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)
        total_image_area = 0

        if images:
            for img in images:
                xref = img[0]
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    for rect in img_rects:
                        image_area = rect.width * rect.height
                        total_image_area += image_area

            page_width, page_height = page.rect.width, page.rect.height
            total_page_area = page_width * page_height
            image_area_ratio = total_image_area / total_page_area

            if image_area_ratio > image_area_threshold:
                image_heavy_pages.append(page_num)

    return image_heavy_pages

def get_gpt4_image_description(image_str):
    """Generates a description of an image using GPT-4."""
    sys_prompt = (
        "You are a software developer and solution architect with a research background. "
        "Provide a detailed description of the image. Consider all the text, screenshots, workflows, "
        "figures, graphs, and captions embedded in the image. The chunk is required for developing a "
        "GenAI-powered RAG. Do not start with 'the image of a...', return the chunk only. Return all content as text."
    )

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"data:image/png;base64,{image_str}"},
        ]
    )

    return response.choices[0].message.content


def convert_images_and_get_gpt4_description(pdf_path, image_heavy_pages, image_folder="image_folder"):
    """Converts image-heavy pages to PNG and uses GPT-4 for descriptions."""
    pdf_document = fitz.open(pdf_path)
    image_descriptions = {}

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    for page_num in image_heavy_pages:
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=150) 
        image_path = os.path.join(image_folder, f"page-{page_num}.png")
        pix.save(image_path)
        

        image_str = encode_image(image_path)
        description = get_gpt4_image_description(image_str)
        image_descriptions[page_num] = description

    return image_descriptions

def process_pdf_for_rag(pdf_path):
    """Main function to process a PDF document for RAG use case."""
    text_chunks = extract_and_chunk_pdf(pdf_path)
    image_heavy_pages = identify_image_heavy_pages(pdf_path)
    image_descriptions = convert_images_and_get_gpt4_description(pdf_path, image_heavy_pages)

    updated_chunks = []
    for chunk in text_chunks:
        if chunk.metadata["page_number"] in image_descriptions:
            description = image_descriptions[chunk.metadata["page_number"]]
            updated_chunk = Document(page_content=description, metadata=chunk.metadata)
            updated_chunks.append(updated_chunk)
        else:
            updated_chunks.append(chunk)

    return updated_chunks

# Example Usage
# if __name__ == "__main__":
#     pdf_path = "/Users/shtlpmac_049/Desktop/chunking-techniques/paper_pdf.pdf"
#     updated_chunks = process_pdf_for_rag(pdf_path)
#     print(updated_chunks)




# from azure.ai.formrecognizer import DocumentAnalysisClient
# from azure.core.credentials import AzureKeyCredential
# from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
# from langchain.docstore.document import Document

# file_path="/Users/shtlpmac_049/Desktop/chunking-techniques/Zomato_2_page.pdf"
# endpoint = "https://shorthills-traning.cognitiveservices.azure.com/"
# key = "bb676acfdba840ee9a30c2e53dac3abc"
# analysis_features = ["ocrHighResolution"]


# loader = AzureAIDocumentIntelligenceLoader(
#         api_endpoint=endpoint,
#         api_key=key,
#         file_path=file_path,
#         api_model="prebuilt-layout",
#         analysis_features=["ocrHighResolution"],
#     )
# # loader = AzureAIDocumentIntelligenceLoader(
# #     api_endpoint=endpoint,
# #     api_key=key,
# #     file_path=file_path,
# #     api_model="prebuilt-layout",
# #     analysis_features=analysis_features,
# # )
# # print(loader)
# # exit()
# # Function to extract text and chunk PDF page-by-page using AzureAIDocumentIntelligenceLoader
# def extract_and_chunk_pdf(path):
#     # Initialize the Azure loader
#     loader = AzureAIDocumentIntelligenceLoader(api_endpoint=endpoint,
#     api_key=key,
#     file_path=file_path,
#     api_model="prebuilt-layout",
#     analysis_features=analysis_features,)

#     # Load the document
#     document = loader.load(path)

#     # Extract data and organize chunks
#     data = document.document.export_to_dict()
#     chunk_list_pagewise = []

#     for page in data.get("pages", []):
#         page_number = page.get("page_number", "Unknown")
#         text = page.get("text", "")

#         # Create a Document chunk with the page content and metadata
#         chunk = Document(page_content=text, metadata={"filename": path, "page_number": page_number})
#         chunk_list_pagewise.append(chunk)

#     return chunk_list_pagewise

# # Function to identify image-heavy pages based on embedded images
# def identify_image_heavy_pages(data, image_area_threshold=0.05):
#     image_heavy_pages = []

#     for page in data.get("pages", []):
#         page_number = page.get("page_number", "Unknown")
#         images = page.get("images", [])

#         if images:
#             total_image_area = 0
#             for img in images:
#                 img_width, img_height = img.get("width", 0), img.get("height", 0)
#                 image_area = img_width * img_height
#                 total_image_area += image_area

#             # Get the total page area (width * height)
#             page_width, page_height = page.get("width", 1), page.get("height", 1)
#             total_page_area = page_width * page_height

#             # Calculate the proportion of the page that is image
#             image_area_ratio = total_image_area / total_page_area

#             # Classify the page as "image-heavy" if the image area exceeds the threshold
#             if image_area_ratio > image_area_threshold:
#                 image_heavy_pages.append(page_number)

#     return image_heavy_pages

# # Function to convert image-heavy pages to PNG and use GPT-4 for analysis
# def convert_images_and_get_gpt4_description(data, image_heavy_pages, image_folder="image_folder"):
#     image_descriptions = {}  # Dictionary to store descriptions for each image-heavy page

#     # Ensure the image folder exists
#     if not os.path.exists(image_folder):
#         os.makedirs(image_folder)

#     for page in data.get("pages", []):
#         page_number = page.get("page_number", "Unknown")

#         if page_number in image_heavy_pages:
#             img_data = page.get("rendered_image")  # Assuming rendered image data is available

#             # Save the page as a PNG file
#             image_path = os.path.join(image_folder, f"page-{page_number}.png")
#             with open(image_path, "wb") as img_file:
#                 img_file.write(img_data)

#             # Send the image to GPT-4 for description
#             description = get_gpt4_image_description(image_path)

#             # Store the description of the image
#             image_descriptions[page_number] = description

#     return image_descriptions

# # Function to generate description from GPT-4
# def get_gpt4_image_description(image_path):
#     # Open the image file in binary format
#     with open(image_path, "rb") as img_file:
#         image_data = img_file.read()

#     # Send the image to GPT-4 (or other image-processing LLMs)
#     response = openai.Image.create(
#         file=image_data,
#         purpose="answers",  # Adjust based on your use case
#         model="gpt-4"  # GPT-4 model or a suitable image analysis model
#     )

#     # Extract textual description from GPT-4's response
#     description = response["choices"][0]["text"]
#     return description

# # Main function to process the PDF document
# def process_pdf_for_rag(pdf_path):
#     # Step 1: Extract text from the PDF and chunk it page-wise
#     chunk_list_pagewise = extract_and_chunk_pdf(pdf_path)

#     # Step 2: Identify image-heavy pages
#     loader = AzureAIDocumentIntelligenceLoader()
#     document = loader.load(pdf_path)
#     data = document.document.export_to_dict()
#     image_heavy_pages = identify_image_heavy_pages(data)

#     # Step 3: Convert image-heavy pages to images and get descriptions from GPT-4
#     image_descriptions = convert_images_and_get_gpt4_description(data, image_heavy_pages)

#     # Step 4: Replace regular text chunks with GPT-4 generated text for image-heavy pages
#     updated_chunks = []
#     for chunk in chunk_list_pagewise:
#         # If the chunk corresponds to an image-heavy page, replace it with the GPT-4 description
#         if chunk.metadata["page_number"] in image_descriptions:
#             description = image_descriptions[chunk.metadata["page_number"]]
#             updated_chunk = Document(page_content=description, metadata=chunk.metadata)
#             updated_chunks.append(updated_chunk)
#         else:
#             updated_chunks.append(chunk)

#     return updated_chunks

# # Example usage
# if __name__ == "__main__":
#     # Set the path to your PDF file
#     pdf_path = "/Users/shtlpmac_049/Desktop/chunking-techniques/Zomato_2_page.pdf"

#     # Process the PDF and generate the chunks
#     updated_chunks = process_pdf_for_rag(pdf_path)

#     # Print the updated chunks
#     for chunk in updated_chunks:
#         print(f"Page {chunk.metadata['page_number']}:\n{chunk.page_content}\n")



# # from docling_parse.pdf_parsers