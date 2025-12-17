import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.chatbot.rag_bot import RAGChatBot
from src.chatbot.prompts import FALLBACK_RESPONSE


class TestRAGChatBot:
    """Unit tests for RAGChatBot class with Google Gemini integration."""

    def test_chatbot_initialization(self, test_settings, mock_embedding_generator, mock_vector_store):
        """Test that RAGChatBot initializes correctly with Gemini config."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                chatbot = RAGChatBot(
                    gemini_config=test_settings.gemini,
                    embedding_generator=mock_embedding_generator,
                    vector_store=mock_vector_store,
                )

                # Verify Gemini is configured
                assert chatbot.gemini_config.model == test_settings.gemini.model
                assert chatbot.gemini_config.api_key == test_settings.gemini.api_key

    def test_ask_with_context(self, test_settings, mock_embedding_generator, mock_vector_store, sample_documents):
        """Test RAG flow with context retrieval and LLM generation."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = MagicMock()
                mock_model.generate_content.return_value.text = "This is a test response."
                mock_model_class.return_value = mock_model

                chatbot = RAGChatBot(
                    gemini_config=test_settings.gemini,
                    embedding_generator=mock_embedding_generator,
                    vector_store=mock_vector_store,
                )

                # Mock embedding and vector store
                mock_embedding_generator.generate_single.return_value = np.random.rand(384)
                mock_docs = [
                    {
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "similarity_score": 0.85
                    }
                    for doc in sample_documents
                ]
                mock_vector_store.search.return_value = mock_docs

                # Call ask
                result = chatbot.ask("What is Python?")

                # Assertions
                assert "answer" in result
                assert "sources" in result
                assert result["model_used"] == test_settings.gemini.model
                mock_embedding_generator.generate_single.assert_called_once()
                mock_vector_store.search.assert_called_once()

    def test_ask_no_context(self, test_settings, mock_embedding_generator, mock_vector_store):
        """Test fallback behavior when no relevant documents are found."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = MagicMock()
                mock_model_class.return_value = mock_model

                chatbot = RAGChatBot(
                    gemini_config=test_settings.gemini,
                    embedding_generator=mock_embedding_generator,
                    vector_store=mock_vector_store,
                )

                mock_embedding_generator.generate_single.return_value = np.random.rand(384)
                mock_vector_store.search.return_value = []

                result = chatbot.ask("Unknown question")

                assert result["answer"] == FALLBACK_RESPONSE
                assert result["sources"] == []

    def test_conversation_history(self, test_settings, mock_embedding_generator, mock_vector_store, sample_documents):
        """Test conversational mode with history management."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = MagicMock()
                mock_model.generate_content.return_value.text = "This is a conversational response."
                mock_model_class.return_value = mock_model

                chatbot = RAGChatBot(
                    gemini_config=test_settings.gemini,
                    embedding_generator=mock_embedding_generator,
                    vector_store=mock_vector_store,
                )

                mock_embedding_generator.generate_single.return_value = np.random.rand(384)
                mock_docs = [{"content": "test", "metadata": {}, "similarity_score": 0.8}]
                mock_vector_store.search.return_value = mock_docs

                history = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]

                result = chatbot.chat("How are you?", history)

                assert "answer" in result
                assert result["answer"] == "This is a conversational response."

    def test_source_attribution(self, test_settings, mock_embedding_generator, mock_vector_store, sample_documents):
        """Test that sources are correctly attributed and formatted."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                chatbot = RAGChatBot(
                    gemini_config=test_settings.gemini,
                    embedding_generator=mock_embedding_generator,
                    vector_store=mock_vector_store,
                )

                mock_embedding_generator.generate_single.return_value = np.random.rand(384)
                mock_docs = [
                    {
                        "content": sample_documents[0]["content"],
                        "metadata": sample_documents[0]["metadata"],
                        "similarity_score": 0.85
                    }
                ]
                mock_vector_store.search.return_value = mock_docs

                sources = chatbot.get_sources("What is Python?")

                assert len(sources) == 1
                assert sources[0]["content"] == sample_documents[0]["content"]
                assert sources[0]["metadata"] == sample_documents[0]["metadata"]
                assert sources[0]["similarity_score"] == 0.85

    def test_get_sources(self, test_settings, mock_embedding_generator, mock_vector_store, sample_documents):
        """Test get_sources method returns correct format."""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                chatbot = RAGChatBot(
                    gemini_config=test_settings.gemini,
                    embedding_generator=mock_embedding_generator,
                    vector_store=mock_vector_store,
                )

                mock_embedding_generator.generate_single.return_value = np.random.rand(384)
                mock_docs = [
                    {
                        "content": sample_documents[0]["content"],
                        "metadata": sample_documents[0]["metadata"],
                        "similarity_score": 0.85
                    }
                ]
                mock_vector_store.search.return_value = mock_docs

                sources = chatbot.get_sources("What is Python?")

                assert len(sources) == 1
                assert sources[0]["content"] == sample_documents[0]["content"]