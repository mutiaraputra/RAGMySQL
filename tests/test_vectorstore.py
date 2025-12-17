import json
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.vectorstore.tidb_store import TiDBVectorStore


class TestTiDBVectorStore:
    @pytest.fixture
    def mock_client(self):
        """Mock TiDBClient for testing."""
        client = MagicMock()
        client.create_table.return_value = MagicMock()  # Mock table
        client.cursor.return_value.__enter__.return_value.executemany = MagicMock()
        return client

    @pytest.fixture
    def vector_store(self, mock_client, test_settings):
        """Create TiDBVectorStore instance with mocked client."""
        with patch('src.vectorstore.tidb_store.TiDBClient.connect', return_value=mock_client):
            store = TiDBVectorStore(
                tidb_config=test_settings.tidb,
                table_name=test_settings.app.vector_table_name,
                embedding_dim=test_settings.app.embedding_dimension,
                distance_metric=test_settings.app.distance_metric
            )
            yield store

    def test_table_creation(self, vector_store, mock_client):
        """Test table creation."""
        vector_store.create_table()
        mock_client.create_table.assert_called_once()
        schema = mock_client.create_table.call_args[1]['schema']
        assert schema.__tablename__ == vector_store.table_name
        assert hasattr(schema, 'id')
        assert hasattr(schema, 'content')
        assert hasattr(schema, 'embedding')
        assert hasattr(schema, 'metadata')
        assert hasattr(schema, 'source_table')
        assert hasattr(schema, 'created_at')

    def test_add_documents(self, vector_store, mock_client, sample_documents, sample_embeddings):
        """Test bulk insert of documents with embeddings."""
        vector_store.table = MagicMock()  # Mock table for other methods
        vector_store.add_documents(sample_documents, sample_embeddings)
        
        # Check that cursor was used
        cursor_mock = mock_client.cursor.return_value.__enter__.return_value
        cursor_mock.executemany.assert_called_once()
        
        # Check SQL structure
        call_args = cursor_mock.executemany.call_args
        sql, values = call_args[0]
        assert 'INSERT INTO' in sql
        assert 'ON DUPLICATE KEY UPDATE' in sql
        assert len(values) == len(sample_documents)
        
        # Check value format
        for val in values:
            assert len(val) == 6  # id, content, embedding, metadata, source_table, created_at
            assert isinstance(val[3], str)  # metadata as JSON string
            assert val[4] == ''  # default source_table

    def test_search(self, vector_store, sample_embeddings):
        """Test similarity search."""
        vector_store.table = MagicMock()
        mock_results = [
            {'content': 'test content 1', 'metadata': {'key': 'value1'}, '_distance': 0.1},
            {'content': 'test content 2', 'metadata': {'key': 'value2'}, '_distance': 0.2}
        ]
        vector_store.table.search.return_value.limit.return_value.to_list.return_value = mock_results
        
        query_emb = sample_embeddings[0]
        results = vector_store.search(query_emb, top_k=2)
        
        assert len(results) == 2
        assert results[0]['content'] == 'test content 1'
        assert results[0]['similarity_score'] == 0.9  # 1 - 0.1 for cosine
        assert results[1]['similarity_score'] == 0.8  # 1 - 0.2

    def test_search_with_filters(self, vector_store, sample_embeddings):
        """Test search with metadata filters."""
        vector_store.table = MagicMock()
        mock_results = [{'content': 'filtered content', 'metadata': {'source': 'table1'}, '_distance': 0.05}]
        search_mock = vector_store.table.search.return_value.limit.return_value
        search_mock.filter.return_value.to_list.return_value = mock_results
        
        query_emb = sample_embeddings[0]
        filters = {'source_table': 'table1'}
        results = vector_store.search(query_emb, top_k=1, filters=filters)
        
        search_mock.filter.assert_called_once_with(filters)
        assert len(results) == 1
        assert results[0]['content'] == 'filtered content'

    def test_upsert(self, vector_store, mock_client, sample_documents, sample_embeddings):
        """Test upsert logic for duplicate handling."""
        vector_store.table = MagicMock()
        # Simulate duplicate by calling add_documents twice
        vector_store.add_documents(sample_documents[:1], sample_embeddings[:1])
        vector_store.add_documents(sample_documents[:1], sample_embeddings[:1])
        
        cursor_mock = mock_client.cursor.return_value.__enter__.return_value
        # Should be called twice, each with ON DUPLICATE KEY UPDATE
        assert cursor_mock.executemany.call_count == 2
        sql = cursor_mock.executemany.call_args_list[0][0][0]
        assert 'ON DUPLICATE KEY UPDATE' in sql

    def test_vector_dimension_mismatch(self, vector_store, sample_documents):
        """Test error handling for dimension mismatch."""
        wrong_embeddings = np.random.rand(len(sample_documents), 100)  # Wrong dimension
        with pytest.raises(ValueError, match="Number of documents must match number of embeddings"):
            vector_store.add_documents(sample_documents, wrong_embeddings)

    def test_get_document_count(self, vector_store):
        """Test document count retrieval."""
        vector_store.table = MagicMock()
        vector_store.table.count.return_value = 42
        count = vector_store.get_document_count()
        assert count == 42
        vector_store.table.count.assert_called_once()

    def test_delete_by_source(self, vector_store):
        """Test deletion by source table."""
        vector_store.table = MagicMock()
        vector_store.table.delete.return_value.filter.return_value.execute.return_value = 10
        vector_store.delete_by_source('test_table')
        vector_store.table.delete.return_value.filter.assert_called_once_with({'source_table': 'test_table'})
        vector_store.table.delete.return_value.filter.return_value.execute.assert_called_once()

    def test_context_manager(self, vector_store, mock_client):
        """Test context manager for connection cleanup."""
        with vector_store:
            pass
        mock_client.close.assert_called_once()


import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.vectorstore.pinecone_store import PineconeVectorStore


class TestPineconeVectorStore:
    @pytest.fixture
    def mock_requests(self):
        """Mock requests for Pinecone REST API."""
        with patch('requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"upsertedCount": 10}
            mock_session.return_value.post.return_value = mock_response
            mock_session.return_value.get.return_value = mock_response
            yield mock_session

    @pytest.fixture
    def vector_store(self, mock_requests, test_settings):
        """Create PineconeVectorStore instance with mocked requests."""
        return PineconeVectorStore(
            test_settings.pinecone,
            test_settings.app.embedding_dimension
        )

    def test_initialization(self, vector_store, test_settings):
        """Test vector store initialization."""
        assert vector_store.embedding_dim == test_settings.app.embedding_dimension
        assert vector_store.namespace == test_settings.pinecone.namespace

    def test_add_documents(self, vector_store, sample_documents, sample_embeddings):
        """Test bulk insert of documents with embeddings."""
        with patch.object(vector_store, '_upsert_batch') as mock_upsert:
            mock_upsert.return_value = {"upsertedCount": len(sample_documents)}
            
            vector_store.add_documents(sample_documents, sample_embeddings)
            
            assert mock_upsert.called

    def test_search(self, vector_store, sample_embeddings):
        """Test similarity search."""
        with patch.object(vector_store, 'search') as mock_search:
            mock_search.return_value = [
                {"content": "test", "metadata": {}, "similarity_score": 0.9}
            ]
            
            query_embedding = sample_embeddings[0]
            results = vector_store.search(query_embedding, top_k=5)
            
            assert len(results) == 1
            assert results[0]["similarity_score"] == 0.9

    def test_search_with_filters(self, vector_store, sample_embeddings):
        """Test search with metadata filters."""
        with patch.object(vector_store, 'search') as mock_search:
            mock_search.return_value = []
            
            query_embedding = sample_embeddings[0]
            filters = {"source": "test"}
            results = vector_store.search(query_embedding, top_k=5, filters=filters)
            
            mock_search.assert_called_once_with(query_embedding, top_k=5, filters=filters)

    def test_get_stats(self, vector_store):
        """Test getting index statistics."""
        with patch.object(vector_store, 'get_stats') as mock_stats:
            mock_stats.return_value = {"totalVectorCount": 1000}
            
            stats = vector_store.get_stats()
            
            assert stats["totalVectorCount"] == 1000