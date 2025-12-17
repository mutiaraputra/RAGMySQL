import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.pipeline.ingestion import IngestionPipeline


class TestIngestionPipeline:
    """Integration tests for IngestionPipeline."""

    @pytest.fixture
    def mock_scraper(self, mocker):
        """Mock JSONScraper."""
        mock = mocker.patch('src.pipeline.ingestion.JSONScraper')
        instance = mock.return_value
        instance.fetch_data.return_value = [
            {"id": 1, "content": "Test document 1"},
            {"id": 2, "content": "Test document 2"}
        ]
        return instance

    @pytest.fixture
    def mock_chunker(self, mocker):
        """Mock TextChunker."""
        mock = mocker.patch('src.pipeline.ingestion.TextChunker')
        instance = mock.return_value
        instance.chunk_text.return_value = [
            {"content": "chunk1", "metadata": {"chunk_index": 0}},
            {"content": "chunk2", "metadata": {"chunk_index": 1}}
        ]
        instance.chunk_documents.return_value = [
            {"content": "chunk1", "metadata": {}},
            {"content": "chunk2", "metadata": {}}
        ]
        return instance

    @pytest.fixture
    def mock_generator(self, mocker):
        """Mock EmbeddingGenerator."""
        mock = mocker.patch('src.pipeline.ingestion.EmbeddingGenerator')
        instance = mock.return_value
        instance.generate_batch.return_value = np.random.rand(2, 384)
        return instance

    @pytest.fixture
    def mock_vectorstore(self, mocker):
        """Mock PineconeVectorStore."""
        mock = mocker.patch('src.pipeline.ingestion.PineconeVectorStore')
        instance = mock.return_value
        instance.add_documents.return_value = None
        instance.get_stats.return_value = {"totalVectorCount": 100}
        return instance

    @pytest.fixture
    def pipeline(self, mock_scraper, mock_chunker, mock_generator, mock_vectorstore, test_settings):
        """Create IngestionPipeline with mocked components."""
        with patch('src.pipeline.ingestion.settings', test_settings):
            pipeline = IngestionPipeline()
            pipeline.scraper = mock_scraper
            pipeline.chunker = mock_chunker
            pipeline.generator = mock_generator
            pipeline.vectorstore = mock_vectorstore
            return pipeline

    def test_full_pipeline_flow(self, pipeline, mock_scraper, mock_chunker, mock_generator, mock_vectorstore, sample_documents):
        """Test end-to-end pipeline flow with mocked components."""
        mock_scraper.fetch_data.return_value = sample_documents
        
        stats = pipeline.run_full_pipeline()
        
        assert 'total_documents' in stats
        assert 'total_chunks' in stats
        assert 'total_embeddings' in stats
        assert 'processing_time' in stats

    def test_pipeline_validation_success(self, pipeline):
        """Test pipeline validation succeeds."""
        result = pipeline.validate_pipeline()
        assert result is True

    def test_checkpoint_save_and_load(self, pipeline, tmp_path):
        """Test checkpoint save and load mechanism."""
        pipeline.checkpoint_file = tmp_path / "test_checkpoint.json"
        
        checkpoint_data = {"table_name": "test_table", "last_id": 100, "processed_docs": 50}
        pipeline._save_checkpoint(**checkpoint_data)
        
        loaded = pipeline._load_checkpoint()
        assert loaded["table_name"] == checkpoint_data["table_name"]
        assert loaded["last_id"] == checkpoint_data["last_id"]

    def test_checkpoint_clear(self, pipeline, tmp_path):
        """Test checkpoint clearing."""
        pipeline.checkpoint_file = tmp_path / "test_checkpoint.json"
        pipeline._save_checkpoint("test", 1, 1)
        assert pipeline.checkpoint_file.exists()
        
        pipeline._clear_checkpoint()
        assert not pipeline.checkpoint_file.exists()