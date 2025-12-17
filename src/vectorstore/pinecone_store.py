import logging
import hashlib
from typing import List, Dict, Any
import requests
import numpy as np
import json

from config.settings import PineconeConfig


class PineconeVectorStore:
    """Vector store using Pinecone Records API (2025-10)."""

    def __init__(self, config: PineconeConfig, embedding_dim: int = 384):
        self.config = config
        self.embedding_dim = embedding_dim
        self.namespace = config.namespace or "default"
        self.batch_size = config.batch_size or 100
        self.logger = logging.getLogger(__name__)
        
        # Setup session
        self.session = requests.Session()
        
        # Base URL from index URL
        self.base_url = str(config.index_url).rstrip('/')
        
        self.logger.info(f"Initialized PineconeVectorStore with Records API (2025-10)")

    def _generate_id(self, content: str, index: int) -> str:
        """Generate a unique ID for a document."""
        hash_input = f"{content[:100]}_{index}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _get_namespace_path(self) -> str:
        """Get namespace path for URL."""
        return "__default__" if self.namespace in ["default", ""] else self.namespace

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray = None) -> None:
        """Add documents to Pinecone using Records API.
        
        Args:
            documents: List of dicts with 'content' and 'metadata' keys
            embeddings: Optional - tidak digunakan karena Pinecone auto-embed
        """
        if len(documents) == 0:
            self.logger.warning("No documents to add")
            return
        
        # Build NDJSON records
        records = []
        for i, doc in enumerate(documents):
            # Generate ID
            doc_id = None
            if 'id' in doc:
                doc_id = str(doc['id'])
            elif 'metadata' in doc and 'id' in doc['metadata']:
                doc_id = str(doc['metadata']['id'])
            elif 'metadata' in doc and 'doc_id' in doc['metadata']:
                doc_id = str(doc['metadata']['doc_id'])
            else:
                doc_id = self._generate_id(doc.get('content', ''), i)
            
            # Get content
            content = doc.get('content', '')
            
            # Build record for Pinecone Records API
            record = {
                "_id": doc_id,
                "text": content
            }
            
            # Add metadata fields
            metadata = doc.get('metadata', {})
            for k, v in metadata.items():
                if v is not None and k not in ['id', 'doc_id', 'content']:
                    if isinstance(v, (str, int, float, bool)):
                        record[k] = v
                    else:
                        record[k] = str(v)
            
            records.append(record)
        
        # Upsert in batches
        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]
            self._upsert_batch(batch)
            self.logger.info(f"Upserted batch {i // self.batch_size + 1}: {len(batch)} records")
        
        self.logger.info(f"Successfully added {len(records)} documents to Pinecone")

    def _upsert_batch(self, records: List[Dict]) -> Dict:
        """Upsert a batch of records to Pinecone using NDJSON format."""
        ns = self._get_namespace_path()
        url = f"{self.base_url}/records/namespaces/{ns}/upsert"
        
        headers = {
            'Api-Key': self.config.api_key,
            'Content-Type': 'application/x-ndjson',
            'X-Pinecone-API-Version': '2025-10',
        }
        
        # Convert to NDJSON (newline-delimited JSON)
        ndjson_data = "\n".join(json.dumps(record) for record in records)
        
        try:
            response = self.session.post(
                url, 
                data=ndjson_data,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            self.logger.debug(f"Upsert response: {response.text}")
            return {"upserted_count": len(records)}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error upserting to Pinecone: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response body: {e.response.text}")
            raise

    def search(self, query_embedding: np.ndarray = None, top_k: int = 5, filters: Dict = None, query_text: str = None) -> List[Dict[str, Any]]:
        """Search for similar documents using Records API.
        
        Args:
            query_embedding: Not used (Pinecone auto-embeds)
            top_k: Number of results to return
            filters: Optional metadata filters
            query_text: Text query for semantic search
            
        Returns:
            List of documents with content, metadata, and similarity_score
        """
        ns = self._get_namespace_path()
        url = f"{self.base_url}/records/namespaces/{ns}/search"
        
        headers = {
            'Accept': 'application/json',
            'Api-Key': self.config.api_key,
            'Content-Type': 'application/json',
            'X-Pinecone-API-Version': '2025-10',
        }
        
        # Build payload sesuai format Pinecone
        payload = {
            "query": {
                "inputs": {
                    "text": query_text or ""
                },
                "top_k": top_k
            }
        }
        
        if filters:
            payload['query']['filter'] = filters
        
        try:
            self.logger.info(f"Searching Pinecone with query: {query_text[:50] if query_text else 'empty'}...")
            
            response = self.session.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            self.logger.debug(f"Search response: {result}")
            
            documents = []
            # Handle response structure
            hits = result.get('result', {}).get('hits', [])
            if not hits:
                hits = result.get('hits', [])
            if not hits:
                hits = result.get('matches', [])
            
            for match in hits:
                fields = match.get('fields', {})
                content = fields.pop('text', '') if 'text' in fields else fields.pop('content', '')
                
                # Get score
                score = match.get('_score', 0.0)
                if not score:
                    score = match.get('score', 0.0)
                
                documents.append({
                    'content': content,
                    'metadata': fields,
                    'similarity_score': score
                })
            
            self.logger.info(f"Found {len(documents)} results")
            return documents
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error searching Pinecone: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response body: {e.response.text}")
            raise

    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using text query (Pinecone auto-embeds)."""
        return self.search(query_text=query, top_k=top_k)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        url = f"{self.base_url}/describe_index_stats"
        
        headers = {
            'Api-Key': self.config.api_key,
            'Content-Type': 'application/json',
        }
        
        try:
            response = self.session.post(url, json={}, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}

    def delete_all(self, namespace: str = None) -> None:
        """Delete all records in namespace."""
        ns = namespace or self._get_namespace_path()
        url = f"{self.base_url}/records/namespaces/{ns}/delete"
        
        headers = {
            'Api-Key': self.config.api_key,
            'Content-Type': 'application/json',
            'X-Pinecone-API-Version': '2025-10',
        }
        
        payload = {'delete_all': True}
        
        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            self.logger.info(f"Deleted all records in namespace: {ns}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error deleting records: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Pinecone vector store context closed")