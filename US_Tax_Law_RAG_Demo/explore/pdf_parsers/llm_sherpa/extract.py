# from llmsherpa.readers import LayoutPDFReader
# llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
# pdf_url = ""
# pdf_reader = LayoutPDFReader(llmsherpa_api_url)
# doc = pdf_reader.read_pdf(pdf_url)

# print(doc[1])



from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

loader = LLMSherpaFileLoader(
    file_path="your pdf path",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="sections",
    llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all%22" # setting up llm sherpa backend service on local
)
docs = loader.load()

print(docs[1])