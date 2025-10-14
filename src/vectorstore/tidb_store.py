import json
import logging
from typing import Dict, List, Optional

import numpy as np
from pydantic import Field
from pytidb import TiDBClient
from pytidb.schema import DistanceMetric, TableModel, VectorField

from config.settings import TiDBConfig

import datetime


class TiDBVectorStore:
    """TiDB Vector Store implementation using pytidb SDK for vector storage and search."""

    def __init__(self, tidb_config: TiDBConfig, table_name: str, embedding_dim: int, distance_metric: str):
        """
        Initialize TiDB Vector Store.

        Args:
            tidb_config: TiDB configuration
            table_name: Name of the vector table
            embedding_dim: Dimension of embedding vectors
            distance_metric: Distance metric ('cosine' or 'L2')
        """
        self.tidb_config = tidb_config
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self.logger = logging.getLogger(__name__)

        # Connect to TiDB
        try:
            self.client = TiDBClient.connect(
                host=tidb_config.host,
                port=tidb_config.port,
                username=tidb_config.user,
                password=tidb_config.password,
                database=tidb_config.database,
                ssl=tidb_config.use_tls
            )
            self.logger.info("Connected to TiDB successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect to TiDB: {e}")
            raise

        # Table model will be created when needed
        self.table = None

    def _get_table_model(self):
        """Dynamically create the TableModel based on configuration."""
        metric = DistanceMetric.COSINE if self.distance_metric == "cosine" else DistanceMetric.L2

        class KnowledgeBase(TableModel):
            __tablename__ = self.table_name
            id: str = Field(primary_key=True)
            content: str = Field()
            embedding: list[float] = VectorField(dimensions=self.embedding_dim, distance_metric=metric)
            metadata: dict = Field(default_factory=dict)
            source_table: str = Field()
            created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

        return KnowledgeBase

    def create_table(self):
        """Create the vector table if it does not exist."""
        try:
            schema = self._get_table_model()
            self.table = self.client.create_table(schema=schema, if_exists="skip")
            self.logger.info(f"Table '{self.table_name}' created or already exists")
        except Exception as e:
            self.logger.error(f"Failed to create table: {e}")
            raise

    def create_vector_index(self):
        """Create vector index on the embedding column. Note: Index is created automatically with VectorField."""
        # The vector index is created automatically when defining VectorField
        # If additional index creation is needed, implement here
        self.logger.info("Vector index creation handled by VectorField definition")

    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """
        Add documents with embeddings to the vector store using upsert logic.

        Args:
            documents: List of document dicts with keys: id, content, metadata, source_table (optional)
            embeddings: Numpy array of embeddings, shape (len(documents), embedding_dim)
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")

        sql = f"""
        INSERT INTO {self.table_name} (id, content, embedding, metadata, source_table, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        content=VALUES(content),
        embedding=VALUES(embedding),
        metadata=VALUES(metadata),
        source_table=VALUES(source_table),
        created_at=VALUES(created_at)
        """

        values = []
        for doc, emb in zip(documents, embeddings):
            values.append((
                doc['id'],
                doc['content'],
                emb.tolist(),
                json.dumps(doc['metadata']),
                doc.get('source_table', ''),
                datetime.datetime.utcnow()
            ))

        try:
            with self.client.cursor() as cursor:
                cursor.executemany(sql, values)
            self.logger.info(f"Inserted/updated {len(documents)} documents")
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise

    def search(self, query_embedding: np.ndarray, top_k: int, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filters: Optional metadata filters

        Returns:
            List of dicts with keys: content, metadata, similarity_score
        """
        if self.table is None:
            raise RuntimeError("Table not created. Call create_table() first.")

        try:
            query_emb = query_embedding.tolist()
            search_query = self.table.search(query_emb).limit(top_k)
            if filters:
                search_query = search_query.filter(filters)
            results = search_query.to_list()

            # Compute similarity score
            search_results = []
            for result in results:
                distance = result['_distance']
                if self.distance_metric == 'cosine':
                    similarity_score = 1 - distance
                else:  # L2
                    similarity_score = 1 / (1 + distance)  # Approximation for L2 similarity
                search_results.append({
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'similarity_score': similarity_score
                })

            self.logger.info(f"Search completed, returned {len(search_results)} results")
            return search_results
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise

    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store."""
        if self.table is None:
            raise RuntimeError("Table not created. Call create_table() first.")
        try:
            count = self.table.count()
            self.logger.info(f"Document count: {count}")
            return count
        except Exception as e:
            self.logger.error(f"Failed to get document count: {e}")
            raise

    def delete_by_source(self, source_table: str):
        """Delete documents by source table."""
        if self.table is None:
            raise RuntimeError("Table not created. Call create_table() first.")
        try:
            deleted_count = self.table.delete().filter({'source_table': source_table}).execute()
            self.logger.info(f"Deleted {deleted_count} documents from source '{source_table}'")
        except Exception as e:
            self.logger.error(f"Failed to delete by source: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        self.logger.info("TiDB connection closed")