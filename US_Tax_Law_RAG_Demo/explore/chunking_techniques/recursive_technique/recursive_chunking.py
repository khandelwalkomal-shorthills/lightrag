import re
from typing import List, Optional, Union


class RecursiveCharacterTextSplitter:
    """Splitting text by recursively looking at characters."""

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: Union[bool, str] = True,
        is_separator_regex: bool = False,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> None:
        """
        Args:
            separators: List of separators to use for splitting text.
            keep_separator: Whether to keep the separator or not in the chunk.
            is_separator_regex: Whether the separators are regular expressions.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._keep_separator = keep_separator
        self._is_separator_regex = is_separator_regex
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def _split_text_with_regex(self, text: str, separator: str) -> List[str]:
        """Helper function to split the text using regular expressions."""
        if self._is_separator_regex:
            return re.split(separator, text)
        else:
            return text.split(separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge smaller pieces into chunks of appropriate size."""
        total = 0
        current_chunk = []
        result = []
        
        for part in splits:
            part_len = len(part)
            
            if total + part_len > self._chunk_size:
                if current_chunk:
                    result.append(separator.join(current_chunk))
                current_chunk = [part]
                total = part_len
            else:
                current_chunk.append(part)
                total += part_len
        
        if current_chunk:
            result.append(separator.join(current_chunk))
        
        return result

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text recursively by checking each separator."""
        final_chunks = []
        separator = separators[-1]
        remaining_separators = separators[:-1]

        # Split the text using the separator
        splits = self._split_text_with_regex(text, separator)
        
        good_splits = []
        
        # For each split, recursively check if it needs further splitting
        for split in splits:
            if len(split) < self._chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if remaining_separators:
                    final_chunks.extend(self._split_text(split, remaining_separators))
                else:
                    final_chunks.append(split)
        
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
        
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into chunks based on defined separators
        and chunk size.
        
        Args:
            text: The input text to be split into chunks.
        
        Returns:
            A list of chunks.
        """
        return self._split_text(text, self._separators)


# ------------------------------
# Demo: Create chunking with sample content
# ------------------------------

def demo_chunking():
    # Sample content to be split
    content = """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Quisque ut venenatis enim. Integer sit amet ex id nulla sollicitudin accumsan.
    Curabitur aliquet tortor sed dui gravida, eget tempus elit gravida.
    
    Vivamus gravida orci at orci sollicitudin, at tincidunt arcu tempus. Ut aliquam purus vel ultricies vehicula.
    Aliquam erat volutpat. Nunc consectetur libero at lectus volutpat, a dapibus risus auctor.
    
    Nullam volutpat, neque sit amet ultricies rhoncus, eros leo bibendum sapien, eu fermentum justo metus at velit.
    Phasellus malesuada erat vitae justo suscipit blandit. Fusce consectetur nisl at fermentum sollicitudin."""

    # Define the splitter with custom separators and chunk size
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        keep_separator=True,
        chunk_size=100,  # Set chunk size to 1000 characters
        chunk_overlap=20  # Set overlap between chunks to 200 characters
    )

    # Split the content into chunks
    chunks = splitter.split_text(content)

    # Print the resulting chunks and metadata
    print("---- Chunks ----")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"Content: {chunk}")
        print("-" * 30)


# Run the demo function
demo_chunking()
