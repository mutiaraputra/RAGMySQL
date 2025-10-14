import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.chunker import TextChunker


class TestEmbeddingGenerator:
    """Unit tests for EmbeddingGenerator class."""

    @pytest.fixture
    def mock_model(self, mocker):
        """Mock SentenceTransformer model for testing."""
        mock_model = mocker.MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        return mock_model

    @pytest.fixture
    def embedding_generator(self, mock_model, mocker):
        """Create EmbeddingGenerator instance with mocked model."""
        with mocker.patch('src.embeddings.generator.SentenceTransformer', return_value=mock_model):
            generator = EmbeddingGenerator(model_name='test-model', device='cpu')
            yield generator

    def test_embedding_generation(self, embedding_generator, mock_model):
        """Test that embeddings are generated with correct dimensions."""
        test_text = "This is a test text."
        mock_embedding = np.random.rand(384)
        mock_model.encode.return_value = mock_embedding

        embedding = embedding_generator.generate_single(test_text)

        assert embedding.shape == (384,)
        assert isinstance(embedding, np.ndarray)
        mock_model.encode.assert_called_once_with(test_text, normalize_embeddings=True)

    def test_embedding_normalization(self, embedding_generator, mock_model):
        """Test that generated embeddings are normalized (unit vectors)."""
        test_text = "Test text for normalization."
        # Create a vector and normalize it
        raw_vector = np.random.rand(384)
        normalized_vector = raw_vector / np.linalg.norm(raw_vector)
        mock_model.encode.return_value = normalized_vector

        embedding = embedding_generator.generate_single(test_text)

        # Check that the vector is normalized (norm ≈ 1)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0), f"Embedding norm is {norm}, expected ≈ 1.0"

    def test_batch_embedding(self, embedding_generator, mock_model):
        """Test batch embedding generation."""
        test_texts = ["Text 1", "Text 2", "Text 3"]
        mock_embeddings = np.random.rand(3, 384)
        mock_model.encode.return_value = mock_embeddings

        embeddings = embedding_generator.generate_embeddings(test_texts)

        assert embeddings.shape == (3, 384)
        assert isinstance(embeddings, np.ndarray)
        mock_model.encode.assert_called_once_with(
            test_texts, 
            normalize_embeddings=True, 
            batch_size=min(100, len(test_texts))  # Assuming default batch_size=100
        )

    def test_embedding_consistency(self, embedding_generator, mock_model):
        """Test that same input produces same embedding."""
        test_text = "Consistent test text."
        mock_embedding = np.random.rand(384)
        mock_model.encode.return_value = mock_embedding

        embedding1 = embedding_generator.generate_single(test_text)
        embedding2 = embedding_generator.generate_single(test_text)

        np.testing.assert_array_equal(embedding1, embedding2)
        # Should only call encode once due to caching
        assert mock_model.encode.call_count == 1

    def test_get_dimension(self, embedding_generator):
        """Test get_dimension method."""
        assert embedding_generator.get_dimension() == 384


class TestTextChunker:
    """Unit tests for TextChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create TextChunker instance."""
        return TextChunker()

    def test_chunker_basic(self, chunker):
        """Test basic text chunking with chunk_size and overlap."""
        text = "This is a long text that should be split into multiple chunks with overlap."
        metadata = {"source": "test", "id": 1}
        
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) > 1  # Should create multiple chunks
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        
        # Check overlap: first chunk should overlap with second
        if len(chunks) > 1:
            first_end = chunks[0]['content'][-50:]  # Last 50 chars of first chunk
            second_start = chunks[1]['content'][:50]  # First 50 chars of second chunk
            # There should be some overlap
            assert len(set(first_end) & set(second_start)) > 0 or first_end in second_start

    def test_chunker_metadata(self, chunker):
        """Test that metadata is preserved and enhanced in chunks."""
        text = "Short text for metadata test."
        metadata = {"source": "test", "id": 1, "extra": "value"}
        
        chunks = chunker.chunk_text(text, metadata)
        
        for i, chunk in enumerate(chunks):
            assert chunk['metadata']['source'] == "test"
            assert chunk['metadata']['id'] == 1
            assert chunk['metadata']['extra'] == "value"
            assert chunk['metadata']['chunk_index'] == i
            assert 'total_chunks' in chunk['metadata']

    def test_chunker_edge_cases(self, chunker):
        """Test edge cases: empty text and very short text."""
        # Empty text
        empty_chunks = chunker.chunk_text("", {"source": "empty"})
        assert empty_chunks == []
        
        # Very short text (shorter than chunk_size)
        short_text = "Hi"
        short_chunks = chunker.chunk_text(short_text, {"source": "short"})
        assert len(short_chunks) == 1
        assert short_chunks[0]['content'] == short_text
        assert short_chunks[0]['metadata']['chunk_index'] == 0
        assert short_chunks[0]['metadata']['total_chunks'] == 1

    def test_chunk_documents(self, chunker):
        """Test batch chunking of multiple documents."""
        documents = [
            {"content": "First document text.", "metadata": {"id": 1}},
            {"content": "Second document text.", "metadata": {"id": 2}}
        ]
        
        all_chunks = chunker.chunk_documents(documents)
        
        assert len(all_chunks) >= 2  # At least one chunk per document
        # Check that chunks from different documents are separated
        chunk_sources = [chunk['metadata']['id'] for chunk in all_chunks]
        assert 1 in chunk_sources
        assert 2 in chunk_sources