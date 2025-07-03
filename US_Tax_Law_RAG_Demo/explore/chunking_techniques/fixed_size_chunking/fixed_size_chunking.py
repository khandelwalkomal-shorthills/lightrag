import re
from typing import List
import tiktoken
ENCODER = None

def word_splitter(source_text: str) -> List[str]:
    """
    Splits a given text into individual words.

    Args:
    - source_text (str): The input text to split.

    Returns:
    - List[str]: A list of words (strings) from the input text.
    """
    # Replace multiple whitespaces with a single space and split by space
    source_text = re.sub(r"\s+", " ", source_text)  # Normalize whitespace
    return re.split(r"\s", source_text)  # Split by whitespace (space, tab, newline)


def get_chunks_fixed_size(text: str, chunk_size: int) -> List[str]:
    """
    Splits the input text into fixed-size chunks based on the number of words.
    Each chunk contains exactly 'chunk_size' words.

    Args:
    - text (str): The input text to be chunked.
    - chunk_size (int): The number of words in each chunk.

    Returns:
    - List[str]: A list of text chunks, each containing 'chunk_size' words.
    """
    # Split the text into words
    text_words = word_splitter(text)

    # Generate the chunks based on the specified chunk size
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        # Select the slice of words for the current chunk
        chunk_words = text_words[i:i + chunk_size]
        # Join the words and add the chunk to the list
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks


# Define overlap as a percent of chunk size
def get_chunks_fixed_size_with_overlap(text: str, chunk_size: int, overlap_fraction: float) -> List[str]:
    """
    Splits the input text into fixed-size chunks with overlap between consecutive chunks.
    The overlap is specified as a fraction of the chunk size.

    Args:
    - text (str): The input text to be chunked.
    - chunk_size (int): The number of words in each chunk.
    - overlap_fraction (float): The fraction of the chunk size that overlaps between consecutive chunks.

    Returns:
    - List[str]: A list of text chunks, each containing 'chunk_size' words, with overlap.
    """
    # Split the text into words
    text_words = word_splitter(text)
    # Calculate the number of words that should overlap
    overlap_size = int(chunk_size * overlap_fraction)

    # Generate the chunks with overlap
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        # Create a chunk with an overlap from the previous chunk
        chunk_words = text_words[max(i - overlap_size, 0): i + chunk_size]
        # Join the words into a chunk and add it to the list
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
        global ENCODER
        if ENCODER is None:
            ENCODER = tiktoken.encoding_for_model(model_name)
        tokens = ENCODER.encode(content)
        return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content
        

#  Used in light rag
def chunking_by_token_size(content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
     ):
        tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
        results = []
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = decode_tokens_by_tiktoken(
                tokens[start : start + max_token_size], model_name=tiktoken_model
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
        print(results)    
        return results


# Example Usage: Fixed-Size Chunking

def chunk_example():
    """
    Demonstrates the usage of fixed-size chunking with different chunk sizes
    and overlap fractions.
    """
    # Fetch the text data (for example, from the Pro Git book)
    url = "https://raw.githubusercontent.com/progit/progit2/main/book/01-introduction/sections/what-is-git.asc"
    import requests
    source_text = requests.get(url).text
    
    chunking_by_token_size(source_text)

    # Chunk text by number of words (with different chunk sizes)
    for chosen_size in [5, 25, 100]:
        # Get chunks without overlap
        print(f"\nChunk size {chosen_size} - Without overlap:")
        chunks = get_chunks_fixed_size(source_text, chosen_size)
        print(f"Total chunks: {len(chunks)}")
        for i in range(min(3, len(chunks))):  # Print first 3 chunks (or all if fewer than 3)
            print(f"Chunk {i + 1}: {chunks[i]}")
        
        # Get chunks with overlap (overlap fraction set to 0.2)
        print(f"\nChunk size {chosen_size} - With overlap (20%):")
        chunks_with_overlap = get_chunks_fixed_size_with_overlap(source_text, chosen_size, 0.2)
        print(f"Total chunks with overlap: {len(chunks_with_overlap)}")
        for i in range(min(3, len(chunks_with_overlap))):  # Print first 3 chunks with overlap
            print(f"Chunk {i + 1}: {chunks_with_overlap[i]}")


# Run the chunking example
if __name__ == "__main__":
    chunk_example()
    # chunking_by_token_size
