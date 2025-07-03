import os
import base64
from pypdf import PdfReader
import fitz  # pymupdf
from langchain.docstore.document import Document
import openai
import json

import pytesseract
from PIL import Image

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
    pdf_document = fitz.open(pdf_path)
    chunk_list_pagewise = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text = page.get_text()  # Extract text
        
        # If no text is found, use OCR
        if not text.strip():
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(image)

        print(f"Page {page_num + 1} Content: {text}")  # Debugging output
        chunk = Document(page_content=text, metadata={"filename": pdf_path, "page_number": page_num})
        chunk_list_pagewise.append(chunk)

    return chunk_list_pagewise

def identify_image_heavy_pages(pdf_path, image_area_threshold=0.80):
    """Identifies image-heavy pages in a PDF based on embedded images."""
    pdf_document = fitz.open(pdf_path)
    image_heavy_pages = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)
        print("Images",images)

        # Debugging: Print details of images
        if not images:
            print(f"No images found on page {page_num + 1}")
        else:
            print(f"Page {page_num + 1} has {len(images)} images. Details:")
            for img in images:
                print(f"Image info: {img}")

        total_image_area = 0
        if images:
            for img in images:
                try:
                    xref = img[0]
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        for rect in img_rects:
                            image_area = rect.width * rect.height
                            total_image_area += image_area
                except Exception as e:
                    print(f"Error processing image on page {page_num + 1}: {e}")

            page_width, page_height = page.rect.width, page.rect.height
        
            total_page_area = page_width * page_height
            
            image_area_ratio = total_image_area / total_page_area
            print("Page ratio", image_area_ratio)

            if image_area_ratio > image_area_threshold:
                image_heavy_pages.append(page_num)
    
    print("*************",image_heavy_pages)
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


def convert_images_and_get_gpt4_description(pdf_path, image_heavy_pages, image_folder="image_folder"):
    """Converts image-heavy pages to PNG and uses GPT-4 for descriptions."""
    pdf_document = fitz.open(pdf_path)
    image_descriptions = {}
    print("*"*10, image_heavy_pages)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    for page_num in image_heavy_pages:
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
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
        page_num = chunk.metadata["page_number"]
        if page_num in image_heavy_pages:
            description = image_descriptions.get(page_num, "").strip()
            updated_chunk = {
                "page_content": description,
                "metadata": {
                    **chunk.metadata,
                    "is_image_heavy_page": True
                }
            }
        else:
            updated_chunk = {
                "page_content": chunk.page_content,
                "metadata": {
                    **chunk.metadata,
                    "is_image_heavy_page": False
                }
            }
        updated_chunks.append(updated_chunk)

    return updated_chunks

if __name__ == "__main__":
    pdf_path = "/Users/shtlpmac_049/Desktop/chunking-techniques/ecg_report.pdf"
    updated_chunks = process_pdf_for_rag(pdf_path)

    # Save JSON output to a file
    output_json_path = "processed_output.json"
    with open(output_json_path, "w") as json_file:
        json.dump(updated_chunks, json_file, indent=4)

    print(f"Processed data saved to {output_json_path}")

    # Optionally, print the JSON output
    print(json.dumps(updated_chunks, indent=4))