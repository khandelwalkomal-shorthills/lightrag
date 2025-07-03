import pdfplumber
import pandas as pd
import os
import json 

folder_path = "your folder path"
pdf_path = "your pdf path"

def process_pdf_with_pdfplumber(pdf_path):
    result = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_data = {}
            raw_content = page.extract_text()  

            # Extract tables
            tables = page.extract_tables()
            table_json = []
            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])  
                table_json.append(df.to_dict(orient="records"))

            page_data["page_no"] = page_num
            page_data["headings"] = []  # Not directly supported
            page_data["sub_headings"] = []  # Not directly supported
            page_data["paragraphs"] = [{"heading": None, "paragraph_content": para.strip()} for para in raw_content.split("\n") if para.strip()]
            page_data["table_content"] = table_json if table_json else None
            page_data["page_content"] = raw_content

            result.append(page_data)
    return result

output = process_pdf_with_pdfplumber(pdf_path)

output_data_path = os.path.join(folder_path, "output_data.json")
with open(output_data_path, "w") as json_file:
    json.dump(output, json_file, indent=4)


