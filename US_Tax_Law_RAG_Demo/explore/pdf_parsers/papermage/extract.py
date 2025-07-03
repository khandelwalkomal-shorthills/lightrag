import papermage
import json
import os

folder_path = "your folder path"
pdf_path = "your pdf path"

def extract_pdf_info(pdf_path):
    document = papermage.Document(pdf_path)
    
    pdf_data = []

    for page_num, page in enumerate(document.pages):  
        page_info = {
            "page no.": page_num + 1,
            "headings": [],
            "paragraphs": [],
            "table content": [],  
            "page_content": page.get_text().strip()  
        }

        headings = find_bold_text(page)
        page_info["headings"] = headings

        paragraphs = extract_paragraphs(page, headings)
        page_info["paragraphs"] = paragraphs

        pdf_data.append(page_info)

    return pdf_data


def find_bold_text(page):
    bold_words = []
    
    for element in page.layout:
        if 'Bold' in element.font_style:
            bold_words.append(element.text)
    
    return bold_words


def extract_paragraphs(page, headings):
    paragraphs = []
    lines = page.get_text().split('\n')

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
