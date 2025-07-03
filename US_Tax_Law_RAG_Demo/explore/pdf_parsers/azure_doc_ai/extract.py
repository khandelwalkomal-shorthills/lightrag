from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

file_path="your pdf path"
endpoint = "your end point"
key = "your azure doc ai key"
analysis_features = ["ocrHighResolution"]
loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint=endpoint,
    api_key=key,
    file_path=file_path,
    api_model="prebuilt-layout",
    analysis_features=analysis_features,
)

documents = loader.load()

print(documents)
