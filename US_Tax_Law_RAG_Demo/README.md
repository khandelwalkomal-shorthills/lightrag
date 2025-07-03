# LightRAG Chatbot Application

This is a chatbot application built using Streamlit and LightRAG that leverages the power of OpenAI's language model. It allows users to interact with a knowledge base loaded from PDF documents.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Load and extract content from PDF files.
- Interact with a chatbot that utilizes a LightRAG knowledge base.
- User-friendly interface built with Streamlit.

## Requirements

- Python 3.10 or higher
- Streamlit
- LightRAG
- OpenAI API Key
- PyPDF2
- python-dotenv

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository:**
   ```bash
   git clone git@github.com:shorthills-ai/US-Tax-Law-RAG-Demo.git
   ```

2. **Create a virtual environment:**
   It’s a good practice to use a virtual environment to manage your dependencies.
   ```bash
   python -m venv light_rag_env
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     light_rag_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source light_rag_env/bin/activate
     ```

4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up your environment variables:**
   Create a `.env` file in the root directory of the project and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. **Run the application:**
- Navigate to light_rag folder and run 
   ```bash
   streamlit run app.py
   ```

2. **Open your web browser:**
   Navigate to `http://localhost:8501` to access the chatbot.

3. **Interact with the chatbot:**
   You can start typing your questions in the chat input box. The chatbot will respond based on the knowledge extracted from the PDF documents.

## Directory Structure

Here’s a brief overview of the directory structure:

```
light_rag
    ├── app.py              # Main application file
    ├── chatbot.py          # Chatbot application logic
    ├── pdf_loader.py       # PDF loading functionality
    ├── knowledge_base.py    # Knowledge base management

── .env                # Environment variables
── requirements.txt     # Python dependencies
── README.md           # Project documentation
```

## Contributing

If you want to contribute to this project, feel free to create a pull request or open an issue for any bugs or feature requests. Contributions are welcome!
