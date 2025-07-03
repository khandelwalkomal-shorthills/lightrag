import json
import os
from fpdf import FPDF
from dotenv import load_dotenv
import openpyxl

class DatasetProcessor:
    def __init__(self):
        """
        Initialize the DataProcessor class, loading the configuration
        from the .env file and setting up necessary variables.
        """
        # Load the environment variables from the .env file
        load_dotenv()
        
        # Get the dataset path from the environment variables
        self.dataset_path = os.getenv('DATASET_PATH', '/Users/shtlpmac_049/Desktop/projects/US-Tax-Law-RAG-main/US-Tax-Law-RAG-Demo/assets/law_dataset.jsonl')  # Set a default if not found
        # self.dataset_path = "/Users/shtlpmac_049/Desktop/projects/US-Tax-Law-RAG-main/US-Tax-Law-RAG-Demo/law_datasett.jsonl"   # Set a default if not found
        
        # Ensure the dataset path is not empty
        if not self.dataset_path:
            raise ValueError("Dataset path is not set in the .env file.")
        
        # Create files for saving extracted data from datasets
        self.output_dir = "../output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")
        
        self.context_file_path = os.path.join(self.output_dir, "output_context.json")
        self.qa_file_path = os.path.join(self.output_dir, "query_and_answer.json")
        self.evaluation_excel_file = os.path.join(self.output_dir, "evaluation_results.xlsx")

        print(f"Dataset path loaded from .env: {self.dataset_path}")

    def process_data(self):
        """
        Parse the JSON data from the input file (line by line) and separate it into two output files:
        1. Full JSON content (containing only 'context' for each record)
        2. Input and answer pairs
        """
        try:
            context_only_data = []
            qa_pairs = []

            # Open the input file and process each line as a separate JSON object
            with open(self.dataset_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    try:
                        # Parse the JSON object from the line
                        record = json.loads(line.strip())
                        
                        # Full JSON data (only 'context' field)
                        context_only_data.append({
                            "context": record.get("context", "")
                        })
                        
                        # Input/Answer pair (input and the first answer)
                        if 'input' in record and 'answers' in record and record['answers']:
                            qa_pairs.append({
                                "input": record['input'],
                                "answer": record['answers'][0]  # Assuming there's only one answer
                            })
                    
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {line.strip()}")
            
            # Write the context-only data to the output file
            with open(self.context_file_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(context_only_data, jsonfile, indent=4, ensure_ascii=False)
            
            # Write the input/answer pairs to the output file
            with open(self.qa_file_path, 'w', encoding='utf-8') as qafile:
                json.dump(qa_pairs, qafile, indent=4, ensure_ascii=False)
            
            print("Data has been parsed and written to the output files successfully.")
        
        except Exception as e:
            print(f"Error occurred during data processing: {e}")

    def get_context_data(self):
        """Reads and returns the context data from the output context file."""
        try:
            with open(self.context_file_path, "r", encoding='utf-8') as data_file:
                context_data = data_file.read()
            return context_data
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {self.context_file_path} was not found.")
        except IOError as e:
            raise IOError(f"An error occurred while reading the file at {self.context_file_path}: {str(e)}")

    def get_query_and_answer(self):
        """Reads and returns the query-answer pairs from the output file."""
        try:
            with open(self.qa_file_path, "r", encoding='utf-8') as data_file:
                qa_data = json.load(data_file)  # Directly parse into a list of dictionaries
            return qa_data
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {self.qa_file_path} was not found.")
        except IOError as e:
            raise IOError(f"An error occurred while reading the file at {self.qa_file_path}: {str(e)}")
        except json.JSONDecodeError:
            raise ValueError(f"The file at {self.qa_file_path} contains invalid JSON.")

    def save_to_excel(self, query, golden_answer, generated_answer, scores):
            """
            Save the query, answers, and scores to an Excel file.
            """
            # Define the Excel file path
            excel_file_path = self.evaluation_excel_file

            # Check if the file exists
            if not os.path.exists(excel_file_path):
                # Create a new workbook and worksheet if the file doesn't exist
                workbook = openpyxl.Workbook()
                sheet = workbook.active
                sheet.title = "Evaluation Results"

                # Write headers
                sheet.append([
                    "Query", "Golden Answer", "Generated Answer", 
                    "Factual Correctness", "Comprehensiveness", "Clarity", "Diversity", "Empowerment", "Overall Correctness"
                ])
            else:
                # Open existing workbook
                workbook = openpyxl.load_workbook(excel_file_path)
                sheet = workbook.active

            # Prepare the data to append to the Excel sheet
            row = [
                query, 
                golden_answer, 
                generated_answer, 
                scores['factual_correctness'], 
                scores['comprehensiveness'], 
                scores['clarity'],
                scores['diversity'],
                scores['empowerment'],
                scores['overall_correctness_percentage']
            ]
            
            # Append the row with new data
            sheet.append(row)

            # Save the updated workbook
            workbook.save(excel_file_path)
            print(f"Results saved to {excel_file_path}") 

# data = DatasetProcessor().process_data()               
