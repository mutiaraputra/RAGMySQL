import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import tqdm

from config.settings import settings
from src.scraper.mysql_scraper import MySQLScraper
from src.embeddings.chunker import TextChunker
from src.embeddings.generator import EmbeddingGenerator
from src.vectorstore.tidb_store import TiDBVectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the full data ingestion pipeline from MySQL to TiDB vector storage."""
    def __init__(self):
        """Initialize all pipeline components."""
        self.scraper = MySQLScraper(settings.mysql)
        self.chunker = TextChunker()
        self.generator = EmbeddingGenerator()
        self.vectorstore = TiDBVectorStore(
            settings.tidb,
            settings.app.vector_table_name,
            settings.app.embedding_dimension,
            settings.app.distance_metric
        )
        self.checkpoint_file = Path("pipeline_checkpoint.json")

    def validate_pipeline(self) -> bool:
        """Validate all connections and configurations before running the pipeline."""
        logger.info("Validating pipeline components...")
        try:
            # Test MySQL connection
            with self.scraper:
                tables = self.scraper.get_tables()
                logger.info(f"MySQL connection valid, found {len(tables)} tables")

            # Test TiDB connection and create table if needed
            self.vectorstore.create_table()
            logger.info("TiDB connection and table creation valid")

            # Validate embedding dimension
            if self.generator.get_dimension() != settings.app.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {settings.app.embedding_dimension}, got {self.generator.get_dimension()}")

            logger.info("Pipeline validation successful")
            return True
        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            raise

    def _save_checkpoint(self, table_name: str, last_id: Optional[int] = None, processed_docs: int = 0):
        """Save pipeline progress to checkpoint file."""
        checkpoint = {
            "table_name": table_name,
            "last_id": last_id,
            "processed_docs": processed_docs,
            "timestamp": time.time()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
        logger.info(f"Checkpoint saved: {checkpoint}")

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load pipeline progress from checkpoint file."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint}")
            return checkpoint
        return None

    def _clear_checkpoint(self):
        """Clear the checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleared")

    def run_full_pipeline(self, table_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full ingestion pipeline in a streaming fashion.

        For each scraped batch: chunk -> embed -> insert, then free memory.

        Args:
            table_config: Dict mapping table_name -> table config (text_columns, id_column)

        Returns:
            Statistics dict with total_documents, total_chunks, total_embeddings, processing_time
        """
        start_time = time.time()
        logger.info("Starting full ingestion pipeline (streaming)")

        # Validate pipeline
        self.validate_pipeline()

        # Checkpoint support (not fully implemented for resume)
        checkpoint = self._load_checkpoint()
        if checkpoint:
            logger.info("Resuming from checkpoint (partial resume not implemented)")

        total_documents = 0
        total_chunks = 0
        total_embeddings = 0

        # Streaming loop: process each scraped batch immediately
        for batch in self.scraper.scrape_all_tables(table_config):
            batch_chunks: List[Dict[str, Any]] = []

            for doc in batch:
                chunks = self.chunker.chunk_text(doc.get('content', ''), doc.get('metadata', {}))

                source_id = doc.get('id') or doc.get('metadata', {}).get('id')
                source_table = doc.get('source_table')
                for chunk in chunks:
                    chunk_index = chunk.get('metadata', {}).get('chunk_index', 0)
                    if source_id is not None:
                        chunk['id'] = f"{source_id}:{chunk_index}"
                    else:
                        chunk['id'] = str(chunk_index)
                    chunk['source_table'] = source_table

                batch_chunks.extend(chunks)
                total_documents += 1

            if not batch_chunks:
                continue

            # Generate embeddings for this batch and insert immediately
            texts = [c['content'] for c in batch_chunks]
            embeddings = self.generator.generate_embeddings(texts)
            self.vectorstore.add_documents(batch_chunks, embeddings)

            total_chunks += len(batch_chunks)
            try:
                total_embeddings += int(embeddings.shape[0])
            except Exception:
                total_embeddings += len(batch_chunks)

            # Free batch memory
            del batch_chunks
            del embeddings

        # Create vector index (if necessary)
        logger.info("Creating vector index (if not present)")
        self.vectorstore.create_vector_index()

        processing_time = time.time() - start_time

        stats = {
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'total_embeddings': total_embeddings,
            'processing_time': processing_time
        }

        logger.info(f"Pipeline completed successfully: {stats}")
        self._clear_checkpoint()
        return stats

    def run_incremental(self, table_name: str, last_id: int) -> Dict[str, Any]:
        """
        Run incremental ingestion for a specific table starting from last_id.

        Args:
            table_name: Name of the table to update
            last_id: Last processed ID to resume from

        Returns:
            Statistics dict
        """
        start_time = time.time()
        logger.info(f"Starting incremental ingestion for table '{table_name}' from ID {last_id}")

        # For incremental, we need to modify the scraper to filter by ID > last_id
        # Since the scraper doesn't support this directly, we'll implement a custom scrape

        total_documents = 0
        all_chunks = []

        # Get text columns from config or assume
        # For simplicity, assume text_columns are provided or fetched
        # This is a placeholder; in real implementation, pass text_columns
        text_columns = ["content"]  # Placeholder, should be from config

        try:
            # Custom incremental scrape
            cursor = self.scraper.connection.cursor(dictionary=True)
            schema = self.scraper.get_table_schema(table_name)
            all_columns = [col["Field"] for col in schema]
            select_columns = ", ".join(all_columns)
            query = f"SELECT {select_columns} FROM {table_name} WHERE id > {last_id} ORDER BY id"
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()

            for row in rows:
                content_parts = [str(row[col]) for col in text_columns if row[col] is not None]
                content = " ".join(content_parts)
                metadata = {col: row[col] for col in all_columns}
                doc = {
                    "id": row["id"],
                    "source_table": table_name,
                    "content": content,
                    "metadata": metadata
                }
                chunks = self.chunker.chunk_text(doc['content'], doc['metadata'])

                # Enrich chunks for incremental ingestion as well
                source_id = doc.get('id') or doc.get('metadata', {}).get('id')
                source_table = doc.get('source_table', table_name)
                for chunk in chunks:
                    chunk_index = chunk.get('metadata', {}).get('chunk_index', 0)
                    if source_id is not None:
                        chunk['id'] = f"{source_id}:{chunk_index}"
                    else:
                        chunk['id'] = str(chunk_index)
                    chunk['source_table'] = source_table

                all_chunks.extend(chunks)
                total_documents += 1

            # Generate embeddings
            texts = [chunk['content'] for chunk in all_chunks]
            embeddings = self.generator.generate_embeddings(texts)

            # Store
            self.vectorstore.add_documents(all_chunks, embeddings)

            processing_time = time.time() - start_time
            stats = {
                'total_documents': total_documents,
                'total_chunks': len(all_chunks),
                'total_embeddings': len(embeddings),
                'processing_time': processing_time
            }

            logger.info(f"Incremental ingestion completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Incremental ingestion failed: {e}")
            raise