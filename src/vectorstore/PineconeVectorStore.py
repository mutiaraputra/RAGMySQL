import json
import logging
from typing import Dict, List, Optional, Any
import requests
import numpy as np

from config.settings import PineconeConfig

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """Pinecone vector store implementation using REST API."""

    def __init__(self, config: PineconeConfig, embedding_dim: int):
        """Initialize Pinecone vector store."""
        self.config = config
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)
        
        self.upsert_url = f"{str(config.index_url).rstrip('/')}/vectors/upsert"
        self.query_url = f"{str(config.index_url).rstrip('/')}/query"
        self.delete_url = f"{str(config.index_url).rstrip('/')}/vectors/delete"
        
        self.headers = {
            'Api-Key': config.api_key,
            'Content-Type': 'application/json'
        }
        
        self.logger.info(f"Initialized PineconeVectorStore with dimension {embedding_dim}")

    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """
        Upsert documents with embeddings to Pinecone.
        
        Args:
            documents: List of dicts with keys: id, content, metadata
            embeddings: Numpy array of embeddings (shape: [n_docs, embedding_dim])
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")
        
        # Process in batches
        batch_size = self.config.batch_size
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            # Prepare vectors for Pinecone
            vectors = []
            for doc, embedding in zip(batch_docs, batch_embeddings):
                vector = {
                    "id": str(doc["id"]),
                    "values": embedding.tolist(),
                    "metadata": {
                        "content": doc["content"][:1000],  # Pinecone metadata size limit
                        **doc.get("metadata", {})
                    }
                }
                vectors.append(vector)
            
            # Prepare request payload
            payload = {"vectors": vectors}
            if self.config.namespace:
                payload["namespace"] = self.config.namespace
            
            # Send upsert request
            try:
                response = requests.post(
                    self.upsert_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                batch_num = (i // batch_size) + 1
                self.logger.info(
                    f"Upserted batch {batch_num}/{total_batches} "
                    f"({len(vectors)} vectors) to Pinecone"
                )
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to upsert batch to Pinecone: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    self.logger.error(f"Response: {e.response.text}")
                raise

    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar vectors in Pinecone.
        
        Args:
            query_embedding: Query vector (shape: [embedding_dim])
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of dicts with keys: id, content, metadata, similarity_score
        """
        try:
            # Prepare query payload
            payload = {
                "vector": query_embedding.tolist(),
                "topK": top_k,
                "includeMetadata": True
            }
            
            if self.config.namespace:
                payload["namespace"] = self.config.namespace
            
            if filters:
                payload["filter"] = filters
            
            # Send query request
            response = requests.post(
                self.query_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            matches = data.get("matches", [])
            
            # Format results
            results = []
            for match in matches:
                result = {
                    "id": match["id"],
                    "content": match.get("metadata", {}).get("content", ""),
                    "metadata": match.get("metadata", {}),
                    "similarity_score": match.get("score", 0.0)
                }
                results.append(result)
            
            self.logger.info(f"Found {len(results)} similar vectors in Pinecone")
            return results
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to query Pinecone: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response: {e.response.text}")
            return []

    def delete_by_ids(self, ids: List[str]):
        """Delete vectors by IDs."""
        try:
            payload = {"ids": ids}
            if self.config.namespace:
                payload["namespace"] = self.config.namespace
            
            response = requests.post(
                self.delete_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            self.logger.info(f"Deleted {len(ids)} vectors from Pinecone")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to delete from Pinecone: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Pinecone vector store context closed")