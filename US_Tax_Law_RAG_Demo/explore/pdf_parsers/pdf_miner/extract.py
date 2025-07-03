from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams, LTTextContainer
import re

import os
import json 

folder_path = "your folder path"
pdf_path = "your pdf path"

def extract_pdf_info(pdf_path):
    raw_text = extract_text(pdf_path, laparams=LAParams())

    pages = raw_text.split('\f')  

    pdf_data = []

    for page_num, page_text in enumerate(pages):
        page_info = {
            "page no.": page_num + 1,
            "headings": [],
            "paragraphs": [],
            "table content": [],  # pdfminer does not support table extraction directly
            "page_content": page_text.strip()
        }

        headings = find_bold_text(page_text)
        page_info["headings"] = headings

        paragraphs = extract_paragraphs(page_text, headings)
        page_info["paragraphs"] = paragraphs

        pdf_data.append(page_info)

    return pdf_data


def find_bold_text(text):
    bold_words = re.findall(r'\b[A-Z][A-Z]+\b', text)
    return bold_words


def extract_paragraphs(text, headings):
    paragraphs = []
    lines = text.split('\n')

    current_paragraph = []
    current_heading = None

    for line in lines:
        if line.strip() in headings:
            if current_paragraph:
                paragraphs.append({"heading": current_heading, "paragraph_content": " ".join(current_paragraph).strip()})
                current_paragraph = []
            current_heading = line.strip()
        else:
            current_paragraph.append(line.strip())

    if current_paragraph:
        paragraphs.append({"heading": current_heading, "paragraph_content": " ".join(current_paragraph).strip()})

    return paragraphs


output = extract_pdf_info(pdf_path)

output_data_path = os.path.join(folder_path, "output_data.json")
with open(output_data_path, "w") as json_file:
    json.dump(output, json_file, indent=4)
