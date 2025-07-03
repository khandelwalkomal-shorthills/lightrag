import boto3
from io import BytesIO
from PyPDF2 import PdfReader
import streamlit as st

class PDFLoader:
    def __init__(self, bucket_name, prefix=''):
        """
        Initialize with S3 bucket name and optional prefix (folder path in bucket)
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3')

    def load_contents(self):
        """Load content from up to 100 PDF files from the specified S3 bucket"""
        pdf_contents = []
        
        try:
            # List objects in the S3 bucket with the given prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )

            # Process up to 5 PDF files
            pdf_count = 0
            for obj in response.get('Contents', []):
                if obj['Key'].lower().endswith('.pdf'):
                    try:
                        # Get the PDF file from S3
                        response = self.s3_client.get_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
                        
                        # Read the PDF content
                        pdf_file = BytesIO(response['Body'].read())
                        reader = PdfReader(pdf_file)
                        text = ''.join(page.extract_text() or '' for page in reader.pages)
                        pdf_contents.append(text.strip())
                        
                        pdf_count += 1
                        if pdf_count >= 500:  # Limit to 5 files as in original code
                            break
                            
                    except Exception as e:
                        st.warning(f"Could not read {obj['Key']}: {e}")
                        
        except Exception as e:
            st.warning(f"Error accessing S3 bucket: {e}")

        return pdf_contents