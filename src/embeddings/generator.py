import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import get_initialized_settings

class EmbeddingGenerator:
    """
    Class for generating text embeddings using SentenceTransformer models.
    
    This class loads a SentenceTransformer model and provides methods to generate
    embeddings for single texts or batches of texts. It includes caching to avoid
    re-computing embeddings for identical texts and supports normalization for
    cosine similarity.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name: Name of the SentenceTransformer model to use. If None, uses the model from AppConfig.
            device: Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        if model_name is None:
            settings = get_initialized_settings()
            model_name = settings.app.embedding_model
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.cache = {}  # Simple cache: text -> embedding
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.logger.info(f"Loaded SentenceTransformer model: {model_name} on device: {device}")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
        
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.logger.info(f"Embedding dimension: {self.dimension}")
    
    def get_dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self.dimension
    
    def generate_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: The input text to embed.
            
        Returns:
            Numpy array of the embedding vector.
        """
        if text in self.cache:
            self.logger.debug("Using cached embedding for text")
            return self.cache[text]
        
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            self.cache[text] = embedding
            self.logger.debug(f"Generated embedding for single text (length: {len(text)})")
            return embedding
        except Exception as e:
            self.logger.error(f"Error encoding single text: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts with internal batching.
        
        Args:
            texts: List of input texts to embed.
            
        Returns:
            Numpy array of shape (len(texts), dimension) containing the embeddings.
        """
        if not texts:
            return np.array([])
        
        # Check cache for each text
        cached_embeddings = []
        uncached_texts = []
        indices = []
        
        for i, text in enumerate(texts):
            if text in self.cache:
                cached_embeddings.append((i, self.cache[text]))
            else:
                uncached_texts.append(text)
                indices.append(i)
        
        # Generate embeddings for uncached texts in batch
        if uncached_texts:
            try:
                settings = get_initialized_settings()
                batch_embeddings = self.model.encode(
                    uncached_texts, 
                    normalize_embeddings=True, 
                    batch_size=min(settings.app.batch_size, len(uncached_texts))
                )
                self.logger.info(f"Generated embeddings for {len(uncached_texts)} texts in batch")
                
                # Cache the new embeddings
                for text, emb in zip(uncached_texts, batch_embeddings):
                    self.cache[text] = emb
            except Exception as e:
                self.logger.error(f"Error encoding batch of texts: {e}")
                raise RuntimeError(f"Batch embedding generation failed: {e}") from e
        else:
            batch_embeddings = np.array([])
        
        # Combine cached and newly generated embeddings in original order
        all_embeddings = [None] * len(texts)
        for i, emb in cached_embeddings:
            all_embeddings[i] = emb
        for idx, emb in zip(indices, batch_embeddings):
            all_embeddings[idx] = emb
        
        return np.array(all_embeddings)