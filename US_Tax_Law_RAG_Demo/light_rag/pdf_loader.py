import os
from PyPDF2 import PdfReader
import streamlit as st

class PDFLoader:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def load_contents(self):
        """Load content from up to 100 PDF files in the specified directory with priority given to certain folders."""
        pdf_contents = []

        def load_pdfs_from_folder(folder_path, limit):
            """Helper function to load PDFs from a specified folder up to a given limit."""
            folder_contents = []
            pdf_count = 0

            for filename in os.listdir(folder_path):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(folder_path, filename)
                    try:
                        with open(pdf_path, 'rb') as file:
                            reader = PdfReader(file)
                            text = ''.join(page.extract_text() or '' for page in reader.pages)
                            folder_contents.append(text.strip())
                            pdf_count += 1
                        
                        if pdf_count >= limit:
                            break
                    except Exception as e:
                        st.warning(f"Could not read {filename}: {e}")

            return folder_contents

        pdf_contents.extend(load_pdfs_from_folder(self.directory_path, 5))

        return pdf_contents
