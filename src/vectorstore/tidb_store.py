import json
import logging
import datetime
from typing import Dict, List, Optional

import numpy as np
import mysql.connector
from mysql.connector import Error
from sklearn.metrics.pairwise import cosine_similarity

from sqlmodel import SQLModel, Field, Column, JSON, create_engine
from typing import Optional, List

from config.settings import TiDBConfig


class TiDBVectorStore:
    """TiDB Vector Store implementation using SQLModel ORM for table creation and data management."""

    def __init__(self, tidb_config: TiDBConfig, table_name: str, embedding_dim: int, distance_metric: str):
        self.tidb_config = tidb_config
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self.logger = logging.getLogger(__name__)
        self.connection = None
        self.client = None
        # --- 1️⃣ Buat koneksi SQLAlchemy engine ---
        try:
            if tidb_config.use_tls:
                self.connection = mysql.connector.connect(
                    host=tidb_config.host,
                    port=tidb_config.port,
                    user=tidb_config.user,
                    password=tidb_config.password,
                    database=tidb_config.database,
                    ssl_verify_cert=True,
                    ssl_ca="certtidb.pem"
                )
            else:
                self.connection = mysql.connector.connect(
                    host=tidb_config.host,
                    port=tidb_config.port,
                    user=tidb_config.user,
                    password=tidb_config.password,
                    database=tidb_config.database
                )

            if self.connection.is_connected():
                self.client = self.connection
                self.logger.info("Connected to TiDB successfully")
            else:
                self.client = None
                raise Exception("Connection object not valid")
        except Error as e:
            self.logger.error(f"Failed to connect to TiDB: {e}")
            self.client = None
            raise

        self.table = None  # Akan diisi oleh _get_table_model

    # --- 2️⃣ Model tabel ---
    def _get_table_model(self):
        table_name = self.table_name

        class KnowledgeBase(SQLModel, table=True):
            __tablename__ = table_name

            id: Optional[int] = Field(default=None, primary_key=True)
            content: str
            meta: Optional[dict] = Field(default=None, sa_column=Column("metadata", JSON))
            vector: Optional[List[float]] = Field(default=None, sa_column=Column(JSON))
            created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

        return KnowledgeBase

    # --- 3️⃣ Buat tabel ---
    def create_table(self):
        try:
            if self.client is None:
                raise RuntimeError("Database connection not established")

            with self.client.cursor() as cursor:
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    content TEXT,
                    metadata JSON,
                    embedding JSON,
                    source_table VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                self.client.commit()
            self.logger.info(f"Table '{self.table_name}' created or already exists")
        except Exception as e:
            self.logger.error(f"Failed to create table: {e}")
            raise

    # --- 4️⃣ Placeholder (index tidak perlu manual di TiDB Vector) ---
    def create_vector_index(self):
        self.logger.info("Vector index creation handled automatically by TiDB.")

    # --- 5️⃣ Tambah dokumen ---
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")

        sql = f"""
        INSERT INTO {self.table_name} (content, embedding, metadata, source_table, created_at)
        VALUES (%s, %s, %s, %s, %s)
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
                doc.get("content"),
                json.dumps(emb.tolist()),
                json.dumps(doc.get("metadata", {})),
                doc.get("source_table"),
                datetime.datetime.utcnow(),
            ))

        try:
            with self.connection.cursor() as cursor:
                cursor.executemany(sql, values)
                self.connection.commit()
            self.logger.info(f"Inserted/updated {len(documents)} documents")
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise

    # --- 6️⃣ Cari dokumen (sementara dummy placeholder) ---
    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute(f"SELECT id, content, embedding, metadata FROM {self.table_name}")
                rows = cursor.fetchall()

            # Jika tidak ada data
            if not rows:
                self.logger.warning("No data in TiDB table to search.")
                return []

            docs = []

            # Normalize query embedding untuk cosine similarity
            query_embedding = query_embedding.reshape(1, -1)

            # Hitung cosine similarity
            for row in rows:
                try:
                    # Parse JSON embedding dari database
                    embedding_str = row["embedding"]
                    if isinstance(embedding_str, str):
                        db_vector = np.array(json.loads(embedding_str))
                    else:
                        db_vector = np.array(embedding_str)
                    
                    # Reshape untuk cosine similarity
                    db_vector = db_vector.reshape(1, -1)
                    
                    # Hitung similarity
                    sim = cosine_similarity(query_embedding, db_vector)[0][0]
                except Exception as e:
                    self.logger.warning(f"Failed to compute similarity for row {row['id']}: {e}")
                    sim = 0.0

                docs.append({
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                    "similarity_score": float(sim)
                })

            # Urutkan berdasarkan similarity tertinggi
            top_docs = sorted(docs, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
            self.logger.info(f"Found {len(top_docs)} documents with similarity scores")
            return top_docs

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    # --- 7️⃣ Tutup koneksi ---
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            try:
                self.client.close()
                self.logger.info("TiDB connection closed")
            except Exception as e:
                self.logger.warning(f"Failed to close TiDB connection: {e}")
