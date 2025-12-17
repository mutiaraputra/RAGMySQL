import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings


class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformers."""

    def __init__(self, model_name: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Use model from settings if not provided
        model_name = model_name or settings.app.embedding_model
        
        # Load model
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.logger.info(f"Loaded SentenceTransformer model: {model_name} on device: {self.model.device}")
        self.logger.info(f"Embedding dimension: {self.dimension}")

    def generate_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            return np.zeros(self.dimension)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def generate_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])
        
        # Filter empty texts and track indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            return np.zeros((len(texts), self.dimension))
        
        # Generate embeddings for valid texts
        self.logger.info(f"Generating embeddings for {len(valid_texts)} texts...")
        embeddings = self.model.encode(
            valid_texts, 
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Create result array with zeros for empty texts
        result = np.zeros((len(texts), self.dimension))
        for idx, embedding in zip(valid_indices, embeddings):
            result[idx] = embedding
        
        self.logger.info(f"Generated {len(embeddings)} embeddings")
        return result

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension