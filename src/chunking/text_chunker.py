import logging
from typing import List, Dict, Any


class TextChunker:
    """Split documents into smaller chunks for embedding."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence/word boundary
            if end < len(text):
                # Look for last period, newline, or space
                for sep in ['. ', '\n', ' ']:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        chunk_text = chunk_text[:last_sep + 1]
                        end = start + last_sep + 1
                        break

            chunk_metadata = dict(metadata) if metadata else {}
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata["start_char"] = start
            chunk_metadata["end_char"] = end

            chunks.append({
                "content": chunk_text.strip(),
                "metadata": chunk_metadata
            })

            # Move start position with overlap
            start = end - self.chunk_overlap
            chunk_index += 1

        self.logger.debug(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents."""
        all_chunks = []

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Add document ID to metadata if available
            if "id" in doc:
                metadata["doc_id"] = doc["id"]

            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)

        self.logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks