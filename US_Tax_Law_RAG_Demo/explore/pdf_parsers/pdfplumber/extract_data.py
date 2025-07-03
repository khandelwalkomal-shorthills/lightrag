import pdfplumber
import json
import os

folder_path = "your folder path"
pdf_path = "your pdf path"

def process_pdf_with_pdfplumber(pdf_path):
    data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_data = {
                "page_no": page_num + 1,
                "headings": [],
                "paragraphs": [],
                "table_content": [],
                "page_content": page.extract_text()
            }

            text = page.extract_text()
            all_words = page.extract_words()

            for word in all_words:
                if word['text'].strip(): 
                    if word.get('doctop') and word.get('top'):
                        if word['top'] < 20: 
                            page_data["headings"].append(word['text'])

            paragraphs = text.split('\n')
            for para in paragraphs:
                if para.strip(): 
                    page_data["paragraphs"].append({
                        "heading": None,  
                        "paragraph_content": para.strip()
                    })

            table = page.extract_tables()
            if table:
                headers = table[0]
                if isinstance(headers, list):
                    headers = [header if isinstance(header, str) else str(header) for header in headers]
                
                for row in table[1:]:
                    if isinstance(row, list): 
                        row_dict = {}
                        for i in range(len(headers)):
                            if i < len(row):
                                row_dict[headers[i]] = row[i] if row[i] is not None else ''
                            else:
                                row_dict[headers[i]] = None
                        page_data["table_content"].append(row_dict)

            data.append(page_data)

    return data


output = process_pdf_with_pdfplumber(pdf_path)

output_data_path = os.path.join(folder_path, "output_extracted_data.json")

with open(output_data_path, "w") as json_file:
    json.dump(output, json_file, indent=4)

print(f"Data extracted and saved to {output_data_path}")
