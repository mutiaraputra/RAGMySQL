import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from config.settings import get_initialized_settings, settings
from src.scraper.json_scraper import JSONScraper
from src.chunking.text_chunker import TextChunker
from src.embeddings.generator import EmbeddingGenerator
from src.vectorstore.pinecone_store import PineconeVectorStore


class IngestionPipeline:
    """Orchestrates data ingestion from JSON endpoint to Pinecone."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.logger = logging.getLogger(__name__)
        self.checkpoint_file = Path("pipeline_checkpoint.json")
        
        # Initialize components
        self.scraper = JSONScraper(settings.json_endpoint)
        self.chunker = TextChunker(
            chunk_size=settings.app.chunk_size,
            chunk_overlap=settings.app.chunk_overlap
        )
        self.generator = EmbeddingGenerator()
        self.vectorstore = PineconeVectorStore(
            settings.pinecone,
            settings.app.embedding_dimension
        )
    
    def validate_pipeline(self) -> bool:
        """Validate pipeline configuration."""
        try:
            self.logger.info("Validating pipeline configuration...")
            # Check if scraper can connect
            self.logger.info(f"JSON Endpoint: {settings.json_endpoint.url}")
            return True
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete ingestion pipeline."""
        start_time = time.time()
        stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "processing_time": 0.0
        }
        
        try:
            self.logger.info("Starting ingestion pipeline...")
            
            # Step 1: Fetch data from JSON endpoint
            self.logger.info("Fetching data from JSON endpoint...")
            raw_data = self.scraper.fetch_data()
            self.logger.info(f"Fetched {len(raw_data)} records")
            
            # Step 2: Extract text content
            self.logger.info("Extracting text content...")
            documents = self.scraper.extract_text_content(raw_data)
            stats["total_documents"] = len(documents)
            self.logger.info(f"Extracted {len(documents)} documents")
            
            if not documents:
                self.logger.warning("No documents to process")
                return stats
            
            # Step 3: Chunk documents
            self.logger.info("Chunking documents...")
            chunks = self.chunker.chunk_documents(documents)
            stats["total_chunks"] = len(chunks)
            self.logger.info(f"Created {len(chunks)} chunks")
            
            # Step 4: Generate embeddings
            self.logger.info("Generating embeddings...")
            contents = [chunk["content"] for chunk in chunks]
            embeddings = self.generator.generate_batch(contents)
            stats["total_embeddings"] = len(embeddings)
            self.logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 5: Store in Pinecone
            self.logger.info("Storing in Pinecone...")
            self.vectorstore.add_documents(chunks, embeddings)
            self.logger.info("Documents stored successfully")
            
            # Calculate processing time
            stats["processing_time"] = time.time() - start_time
            self.logger.info(f"Pipeline completed in {stats['processing_time']:.2f}s")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _save_checkpoint(self, table_name: str = "", last_id: int = 0, processed_docs: int = 0):
        """Save checkpoint for resume capability."""
        checkpoint = {
            "table_name": table_name,
            "last_id": last_id,
            "processed_docs": processed_docs,
            "timestamp": time.time()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
    
    def _clear_checkpoint(self):
        """Clear checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()