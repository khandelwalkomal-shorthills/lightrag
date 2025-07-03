from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.

Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.

It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer. [1]
"""


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 65, chunk_overlap=0)

text_splitter.create_documents([text])


# Above out put will be like below

"""
[Document(metadata={}, page_content="One of the most important things I didn't understand about the"),
 Document(metadata={}, page_content='world when I was a child is the degree to which the returns for'), 
 Document(metadata={}, page_content='performance are superlinear.'), Document(metadata={}, page_content='Teachers and coaches implicitly told us the returns were linear.'), 
 Document(metadata={}, page_content='"You get out," I heard a thousand times, "what you put in." They'), 
 Document(metadata={}, page_content='meant well, but this is rarely true. If your product is only'), 
 Document(metadata={}, page_content="half as good as your competitor's, you don't get half as many"), 
 Document(metadata={}, page_content='customers. You get no customers, and you go out of business.'), 
 Document(metadata={}, page_content="It's obviously true that the returns for performance are"), 
 Document(metadata={}, page_content='superlinear in business. Some think this is a flaw of'), 
 Document(metadata={}, page_content='capitalism, and that if we changed the rules it would stop being'), 
 Document(metadata={}, page_content='true. But superlinear returns for performance are a feature of'), 
 Document(metadata={}, page_content="the world, not an artifact of rules we've invented. We see the"), 
 Document(metadata={}, page_content='same pattern in fame, power, military victories, knowledge, and'), 
 Document(metadata={}, page_content='even benefit to humanity. In all of these, the rich get richer.'), 
 Document(metadata={}, page_content='[1]')]

"""