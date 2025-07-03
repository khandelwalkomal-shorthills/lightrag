import os
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from pi_heif import register_heif_opener
import pytesseract
import json
import pdfplumber

os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.tesseract_ocr.OCRAgentTesseract"

# Ensure the OCR agent is set to Tesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"  # Update this path if necessary


# os.environ["OCR_AGENT"] = "tesseract"
# # Specify the path to your PDF file
# filename = "/Users/shtlpmac_049/Downloads/salesforce-fy24-annual-report.pdf"

# Function to extract and partition content from a PDF
def extract_and_partition_pdf(filename):
    # Extract elements from the PDF using partition_pdf
    elements = partition_pdf(
        filename=filename,
        strategy="hi_res",  # Strategy for high resolution partitioning
        infer_table_structure=True,  # Automatically infers table structure
        model_name="yolox"  # YOLOX model for table extraction
    )

    # Convert extracted elements to JSON format (optional, depending on the use case)
    elements_json = elements_to_json(elements)

    return elements_json


def extract_and_partition_pdf1(filename):
    # Open the PDF file using pdfplumber
    with pdfplumber.open(filename) as pdf:
        extracted_elements = []
        
        for page_number, page in enumerate(pdf.pages, start=1):
            # Extract the text content from the page
            page_text = page.extract_text()
            
            # Extract tables from the page using pdfplumber's table extraction
            tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})

            # Process tables to ensure proper format and structure
            processed_tables = []
            for table in tables:
                processed_table = []
                for row in table:
                    processed_table.append(row)  # Handle rows and columns properly
                processed_tables.append(processed_table)
            
            # Store text content and tables for the page
            page_elements = {
                "page_number": page_number,
                "text": page_text,
                "tables": processed_tables
            }
            extracted_elements.append(page_elements)
    
    # Convert the extracted elements to JSON format
    elements_json = json.dumps(extracted_elements, indent=4)
    
    return elements_json



# def extract_and_partition_pdf(filepath):
#         df_elements = partition_pdf(
#             filename=filepath,
            
#             # Using pdf format to find embedded image blocks
#             extract_images_in_pdf=True,
            
#             # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
#             # Titles are any sub-section of the document
#             infer_table_structure=True,
            
#             # Post processing to aggregate text once we have the title
#             chunking_strategy="by_title",
#             # Chunking params to aggregate text blocks
#             # Attempt to create a new chunk 3800 chars
#             # Attempt to keep chunks > 2000 chars
#             # Hard max on chunks
#             max_characters=4000,
#             new_after_n_chars=3800,
#             combine_text_under_n_chars=2000,
#             image_output_dir_path="..static/pdfImages/",
#         )
#         elements_json = elements_to_json(df_elements)

#         return elements_json

# Example usage:
if __name__ == "__main__":
    # pdf_file = "/Users/shtlpmac_049/Downloads/salesforce-65_70.pdf"
    pdf_file = "/Users/shtlpmac_049/Desktop/chunking-techniques/multiplication-tables-from-1-to-30.pdf"
    # pdf_file = "/Users/shtlpmac_049/Desktop/chunking-techniques/tr_opt_02.pmg.pdf"
    extracted_data = extract_and_partition_pdf(pdf_file)
    # extracted_data = extract_and_partition_pdf1(pdf_file)
    
    # Output the extracted data in JSON format
    print(extracted_data)


    """
     use dependency : poppler (save the envirement path)

    
    
    
    
    """
