import fitz 
import os
import json
import tabula  
import camelot 

folder_path = "your folder path"
pdf_path = "your pdf path"

def process_pdf_with_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    output = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_data = {}

        page_data["page no."] = page_num + 1

        page_data["headings"] = []
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block['type'] == 0:  
                for line in block["lines"]:
                    for span in line["spans"]:
                        if "bold" in span['font'].lower(): 
                            page_data["headings"].append(span['text'])

        paragraphs = page.get_text("text").split("\n")
        page_data["paragraphs"] = []
        for para in paragraphs:
            para_data = {
                "heading": None, 
                "paragraph_content": para
            }
            if para in page_data["headings"]:
                para_data["heading"] = para
            page_data["paragraphs"].append(para_data)

        page_table_data = []
        try:
            tabula_tables = tabula.read_pdf(pdf_path, pages=page_num + 1, multiple_tables=True, lattice=True, guess=True)
            if tabula_tables:
                for table in tabula_tables:
                    table_json = table.to_dict(orient="records")
                    page_table_data.append(table_json)
        except Exception as e:
            print(f"Error using Tabula on page {page_num + 1}: {e}")
        
        if not page_table_data:
            try:
                camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='stream', strip_text='\n')
                if camelot_tables:
                    for table in camelot_tables:
                        table_json = table.df.to_dict(orient="records")
                        page_table_data.append(table_json)
            except Exception as e:
                print(f"Error using Camelot on page {page_num + 1}: {e}")

        if page_table_data:
            page_data["table content"] = page_table_data
        else:
            page_data["table content"] = {}

        page_data["page_content"] = page.get_text("text")

        output.append(page_data)

    return output


output = process_pdf_with_pymupdf(pdf_path)

output_data_path = os.path.join(folder_path, "output_tablu_camelot_data.json")
with open(output_data_path, "w") as json_file:
    json.dump(output, json_file, indent=4)
