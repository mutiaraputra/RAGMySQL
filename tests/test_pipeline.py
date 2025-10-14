import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from src.pipeline.ingestion import IngestionPipeline


class TestIngestionPipeline:
    """Integration tests for IngestionPipeline."""

    @pytest.fixture
    def mock_scraper(self, mocker):
        """Mock MySQLScraper."""
        mock = mocker.MagicMock()
        mock.get_tables.return_value = ['table1', 'table2']
        return mock

    @pytest.fixture
    def mock_chunker(self, mocker):
        """Mock TextChunker."""
        mock = mocker.MagicMock()
        mock.chunk_text.return_value = [
            {'content': 'chunk1', 'metadata': {'id': 1}},
            {'content': 'chunk2', 'metadata': {'id': 1}}
        ]
        return mock

    @pytest.fixture
    def mock_generator(self, mocker):
        """Mock EmbeddingGenerator."""
        mock = mocker.MagicMock()
        mock.get_dimension.return_value = 384
        mock.generate_embeddings.return_value = [[0.1] * 384, [0.2] * 384]  # Mock embeddings
        return mock

    @pytest.fixture
    def mock_vectorstore(self, mocker):
        """Mock TiDBVectorStore."""
        mock = mocker.MagicMock()
        return mock

    @pytest.fixture
    def pipeline(self, mock_scraper, mock_chunker, mock_generator, mock_vectorstore, test_settings):
        """Create IngestionPipeline with mocked components."""
        with patch('src.pipeline.ingestion.MySQLScraper', return_value=mock_scraper), \
             patch('src.pipeline.ingestion.TextChunker', return_value=mock_chunker), \
             patch('src.pipeline.ingestion.EmbeddingGenerator', return_value=mock_generator), \
             patch('src.pipeline.ingestion.TiDBVectorStore', return_value=mock_vectorstore), \
             patch('src.pipeline.ingestion.settings', test_settings):
            pipeline = IngestionPipeline()
            return pipeline

    def test_full_pipeline_flow(self, pipeline, mock_scraper, mock_chunker, mock_generator, mock_vectorstore, sample_documents):
        """Test end-to-end pipeline flow with mocked components."""
        table_config = {'table1': ['content']}
        
        # Mock scrape_all_tables to yield batches
        mock_scraper.scrape_all_tables.return_value = [sample_documents]
        
        # Run pipeline
        stats = pipeline.run_full_pipeline(table_config)
        
        # Assert methods called in order
        mock_scraper.scrape_all_tables.assert_called_once_with(table_config)
        assert mock_chunker.chunk_text.call_count == len(sample_documents)
        mock_generator.generate_embeddings.assert_called_once()
        mock_vectorstore.add_documents.assert_called_once()
        mock_vectorstore.create_vector_index.assert_called_once()
        
        # Assert stats
        assert 'total_documents' in stats
        assert 'total_chunks' in stats
        assert 'total_embeddings' in stats
        assert 'processing_time' in stats
        assert stats['total_documents'] == len(sample_documents)
        assert stats['total_chunks'] == len(sample_documents) * 2  # Assuming 2 chunks per doc
        assert stats['total_embeddings'] == len(sample_documents) * 2

    def test_pipeline_validation_success(self, pipeline, mock_scraper, mock_chunker, mock_generator, mock_vectorstore):
        """Test pipeline validation succeeds."""
        mock_scraper.__enter__ = MagicMock(return_value=mock_scraper)
        mock_scraper.__exit__ = MagicMock(return_value=None)
        
        result = pipeline.validate_pipeline()
        assert result is True
        mock_scraper.get_tables.assert_called_once()
        mock_vectorstore.create_table.assert_called_once()
        mock_generator.get_dimension.assert_called_once()

    def test_pipeline_validation_failure_dimension_mismatch(self, pipeline, mock_generator):
        """Test pipeline validation fails on dimension mismatch."""
        mock_generator.get_dimension.return_value = 512  # Different from config
        
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            pipeline.validate_pipeline()

    def test_pipeline_validation_failure_connection(self, pipeline, mock_scraper):
        """Test pipeline validation fails on connection error."""
        mock_scraper.get_tables.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            pipeline.validate_pipeline()

    def test_pipeline_error_recovery_scrape_failure(self, pipeline, mock_scraper):
        """Test error handling during scraping."""
        table_config = {'table1': ['content']}
        mock_scraper.scrape_all_tables.side_effect = Exception("Scrape failed")
        
        with pytest.raises(Exception, match="Scrape failed"):
            pipeline.run_full_pipeline(table_config)

    def test_pipeline_statistics(self, pipeline, mock_scraper, mock_chunker, mock_generator, mock_vectorstore, sample_documents):
        """Test that pipeline returns correct statistics."""
        table_config = {'table1': ['content']}
        mock_scraper.scrape_all_tables.return_value = [sample_documents]
        
        stats = pipeline.run_full_pipeline(table_config)
        
        assert isinstance(stats, dict)
        assert all(key in stats for key in ['total_documents', 'total_chunks', 'total_embeddings', 'processing_time'])
        assert stats['processing_time'] > 0

    def test_checkpoint_save_and_load(self, pipeline, tmp_path):
        """Test checkpoint save and load mechanism."""
        pipeline.checkpoint_file = tmp_path / "test_checkpoint.json"
        
        checkpoint_data = {"table_name": "test_table", "last_id": 100, "processed_docs": 50}
        pipeline._save_checkpoint(**checkpoint_data)
        
        loaded = pipeline._load_checkpoint()
        assert loaded == checkpoint_data

    def test_checkpoint_clear(self, pipeline, tmp_path):
        """Test checkpoint clearing."""
        pipeline.checkpoint_file = tmp_path / "test_checkpoint.json"
        pipeline._save_checkpoint("test", 1, 1)
        assert pipeline.checkpoint_file.exists()
        
        pipeline._clear_checkpoint()
        assert not pipeline.checkpoint_file.exists()

    def test_run_incremental(self, pipeline, mock_chunker, mock_generator, mock_vectorstore, sample_documents):
        """Test incremental ingestion."""
        # Mock the scraper's connection and cursor
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = sample_documents
        pipeline.scraper.connection.cursor.return_value = mock_cursor
        pipeline.scraper.get_table_schema.return_value = [{"Field": "id"}, {"Field": "content"}]
        
        stats = pipeline.run_incremental("test_table", 0)
        
        assert 'total_documents' in stats
        assert stats['total_documents'] == len(sample_documents)
        mock_cursor.execute.assert_called_once()
        mock_generator.generate_embeddings.assert_called_once()
        mock_vectorstore.add_documents.assert_called_once()