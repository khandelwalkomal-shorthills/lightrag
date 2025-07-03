from langchain.text_splitter import MarkdownTextSplitter
splitter = MarkdownTextSplitter(chunk_size = 40, chunk_overlap=0)

markdown_text = """
# Fun in California

## Driving

Try driving on the 1 down to San Diego

### Food

Make sure to eat a burrito while you're there

## Hiking

Go to Yosemite
"""

print(splitter.create_documents([markdown_text]))

"""   
    *********  output  ********** 

    [Document(metadata={}, page_content='# Fun in California\n\n## Driving'), 
    Document(metadata={}, page_content='Try driving on the 1 down to San Diego'), 
    Document(metadata={}, page_content='### Food'), 
    Document(metadata={}, page_content="Make sure to eat a burrito while you're"), 
    Document(metadata={}, page_content='there'), 
    Document(metadata={}, page_content='## Hiking\n\nGo to Yosemite')]






"""
