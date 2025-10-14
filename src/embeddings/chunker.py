import logging
from typing import Dict, List, Any

from config.settings import settings

logger = logging.getLogger(__name__)


class TextChunker:
    """Class for splitting long texts into overlapping chunks for embedding generation."""

    def __init__(self):
        """Initialize the TextChunker with configuration from settings."""
        self.chunk_size = settings.app.chunk_size
        self.chunk_overlap = settings.app.chunk_overlap

        # Validate configuration
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a single text into chunks with overlap.

        Args:
            text: The text to chunk.
            metadata: Metadata dictionary associated with the text.

        Returns:
            List of dictionaries with 'content' and 'metadata' keys.
        """
        if not text:
            logger.warning("Empty text provided, returning empty chunks")
            return []

        chunks = []
        start = 0
        text_len = len(text)
        chunk_index = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_content = text[start:end]
            chunk_metadata = {
                **metadata,
                'chunk_index': chunk_index,
                'total_chunks': None,  # Will be set after all chunks are created
            }
            chunks.append({
                'content': chunk_content,
                'metadata': chunk_metadata
            })
            chunk_index += 1
            start += self.chunk_size - self.chunk_overlap

        # Set total_chunks in metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = total_chunks

        # Log statistics
        avg_chunk_size = sum(len(c['content']) for c in chunks) / total_chunks if total_chunks > 0 else 0
        logger.info(f"Created {total_chunks} chunks from text, average chunk size: {avg_chunk_size:.1f} characters")

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents in batch.

        Args:
            documents: List of documents, each with 'content' and 'metadata' keys.

        Returns:
            List of chunked documents.
        """
        all_chunks = []
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)

        logger.info(f"Total chunks created from {len(documents)} documents: {len(all_chunks)}")
        return all_chunks