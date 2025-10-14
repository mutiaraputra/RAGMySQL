import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.chatbot.rag_bot import RAGChatBot
from src.chatbot.prompts import FALLBACK_RESPONSE


class TestRAGChatBot:
    """Unit tests for RAGChatBot class with OpenRouter integration."""

    def test_chatbot_initialization(self, test_settings, mock_embedding_generator, mock_vector_store, mock_openrouter_client):
        """Test that RAGChatBot initializes correctly with OpenRouter config."""
        chatbot = RAGChatBot(
            openrouter_config=test_settings.openrouter,
            embedding_generator=mock_embedding_generator,
            vector_store=mock_vector_store,
        )

        # Verify ChatOpenAI is initialized with correct parameters (provided by our ChatOpenAI factory)
        assert chatbot.llm.model == test_settings.openrouter.model
        assert chatbot.llm.api_key == test_settings.openrouter.api_key
        assert str(chatbot.llm.base_url) == str(test_settings.openrouter.base_url)
        # get_openrouter_headers is a method on Settings which uses openrouter attr
        assert chatbot.llm.default_headers == test_settings.openrouter.headers()

        # Verify base_url is correct for OpenRouter
        assert str(chatbot.llm.base_url) == "https://openrouter.ai/api/v1"

    def test_ask_with_context(self, test_settings, mock_embedding_generator, mock_vector_store, sample_documents, mock_openrouter_client):
        """Test RAG flow with context retrieval and LLM generation."""
        chatbot = RAGChatBot(
            openrouter_config=test_settings.openrouter,
            embedding_generator=mock_embedding_generator,
            vector_store=mock_vector_store,
        )

        # Mock the embedding generation
        mock_embedding_generator.generate_single.return_value = np.random.rand(384)

        # Mock vector store search to return sample documents
        mock_docs = [
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity_score": 0.9 - i * 0.1  # Decreasing scores
            }
            for i, doc in enumerate(sample_documents[:3])
        ]
        mock_vector_store.search.return_value = mock_docs

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "This is a generated answer based on the context."
        chatbot.llm.invoke.return_value = mock_response

        # Call ask method
        result = chatbot.ask("What is machine learning?", top_k=3)

        # Assertions
        assert "answer" in result
        assert "sources" in result
        assert "model_used" in result
        assert "confidence_score" in result

        assert result["answer"] == "This is a generated answer based on the context."
        assert result["model_used"] == test_settings.openrouter.model
        assert len(result["sources"]) == 3
        assert result["confidence_score"] > 0

        # Verify sources structure
        for source in result["sources"]:
            assert "content" in source
            assert "metadata" in source
            assert "similarity_score" in source

        # Verify mocks were called
        mock_embedding_generator.generate_single.assert_called_once_with("What is machine learning?")
        mock_vector_store.search.assert_called_once()
        chatbot.llm.invoke.assert_called_once()

    def test_ask_no_context(self, test_settings, mock_embedding_generator, mock_vector_store, mock_openrouter_client):
        """Test fallback behavior when no relevant documents are found."""
        chatbot = RAGChatBot(
            openrouter_config=test_settings.openrouter,
            embedding_generator=mock_embedding_generator,
            vector_store=mock_vector_store,
        )

        # Mock embedding generation
        mock_embedding_generator.generate_single.return_value = np.random.rand(384)

        # Mock vector store to return no documents
        mock_vector_store.search.return_value = []

        # Call ask method
        result = chatbot.ask("What is quantum physics?")

        # Assertions
        assert result["answer"] == FALLBACK_RESPONSE
        assert result["sources"] == []
        assert result["confidence_score"] == 0.0
        assert result["model_used"] == test_settings.openrouter.model

        # Verify LLM was not called since no context
        chatbot.llm.invoke.assert_not_called()

    def test_conversation_history(self, test_settings, mock_embedding_generator, mock_vector_store, sample_documents, mock_openrouter_client):
        """Test conversational mode with history management."""
        chatbot = RAGChatBot(
            openrouter_config=test_settings.openrouter,
            embedding_generator=mock_embedding_generator,
            vector_store=mock_vector_store,
        )

        # Mock embedding and vector store
        mock_embedding_generator.generate_single.return_value = np.random.rand(384)
        mock_docs = [
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity_score": 0.8
            }
            for doc in sample_documents[:2]
        ]
        mock_vector_store.search.return_value = mock_docs

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "This is a conversational response."
        chatbot.llm.invoke.return_value = mock_response

        # Initial conversation history
        history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]

        # Call chat method
        result = chatbot.chat("Tell me more about AI", history)

        # Assertions
        assert result["answer"] == "This is a conversational response."
        assert len(result["sources"]) == 2
        assert result["model_used"] == test_settings.openrouter.model

        # Verify history was updated
        assert len(history) == 4  # Original 2 + user + assistant
        assert history[-2]["role"] == "user"
        assert history[-2]["content"] == "Tell me more about AI"
        assert history[-1]["role"] == "assistant"
        assert history[-1]["content"] == "This is a conversational response."

    def test_source_attribution(self, test_settings, mock_embedding_generator, mock_vector_store, sample_documents, mock_openrouter_client):
        """Test that sources are correctly attributed and formatted."""
        chatbot = RAGChatBot(
            openrouter_config=test_settings.openrouter,
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
            for doc in sample_documents[:1]
        ]
        mock_vector_store.search.return_value = mock_docs

        # Call get_sources method
        sources = chatbot.get_sources("What is Python?")

        # Assertions
        assert len(sources) == 1
        assert sources[0]["content"] == sample_documents[0]["content"]
        assert sources[0]["metadata"] == sample_documents[0]["metadata"]
        assert sources[0]["similarity_score"] == 0.85

    def test_openrouter_configuration(self, test_settings, mock_embedding_generator, mock_vector_store, mock_openrouter_client):
        """Test that OpenRouter configuration is correctly applied."""
        chatbot = RAGChatBot(
            openrouter_config=test_settings.openrouter,
            embedding_generator=mock_embedding_generator,
            vector_store=mock_vector_store,
        )

        # Verify base URL
        assert str(chatbot.llm.base_url) == "https://openrouter.ai/api/v1"

        # Verify headers include attribution if configured
        headers = chatbot.llm.default_headers
        if test_settings.openrouter.app_url:
            assert "HTTP-Referer" in headers
            assert headers["HTTP-Referer"] == test_settings.openrouter.app_url
        if test_settings.openrouter.app_name:
            assert "X-Title" in headers
            assert headers["X-Title"] == test_settings.openrouter.app_name

    def test_openrouter_api_error_handling(self, test_settings, mock_embedding_generator, mock_vector_store):
        """Test error handling for OpenRouter API failures when ChatOpenAI initialization fails."""
        # Patch the ChatOpenAI symbol in the rag_bot module to raise on construction
        from unittest.mock import patch
        with patch('src.chatbot.rag_bot.ChatOpenAI', side_effect=Exception("OpenRouter API rate limit exceeded")):
            # Constructing RAGChatBot should raise or handle the exception when calling ask
            chatbot = RAGChatBot(
                openrouter_config=test_settings.openrouter,
                embedding_generator=mock_embedding_generator,
                vector_store=mock_vector_store,
            )

            # Mock embedding and vector store for successful retrieval
            mock_embedding_generator.generate_single.return_value = np.random.rand(384)
            mock_vector_store.search.return_value = [
                {"content": "Some content", "metadata": {}, "similarity_score": 0.8}
            ]

            # Call ask method - should handle the error gracefully
            result = chatbot.ask("Test question")

            # Assertions for error response
            assert "error occurred" in result["answer"].lower()
            assert result["sources"] == []
            assert result["confidence_score"] == 0.0
            assert result["model_used"] == test_settings.openrouter.model

    def test_chat_error_handling(self, test_settings, mock_embedding_generator, mock_vector_store):
        """Test error handling in conversational mode."""
        chatbot = RAGChatBot(
            openrouter_config=test_settings.openrouter,
            embedding_generator=mock_embedding_generator,
            vector_store=mock_vector_store,
        )

        # Mock embedding to raise exception
        mock_embedding_generator.generate_single.side_effect = Exception("Embedding generation failed")

        history = []
        result = chatbot.chat("Test question", history)

        # Assertions
        assert "error occurred" in result["answer"].lower()
        assert result["sources"] == []
        assert result["confidence_score"] == 0.0
        mock_docs = [
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity_score": 0.85
            }
            for doc in sample_documents[:1]
        ]
        mock_vector_store.search.return_value = mock_docs

        # Call get_sources method
        sources = chatbot.get_sources("What is Python?")

        # Assertions
        assert len(sources) == 1
        assert sources[0]["content"] == sample_documents[0]["content"]
        assert sources[0]["metadata"] == sample_documents[0]["metadata"]
        assert sources[0]["similarity_score"] == 0.85

    def test_openrouter_configuration(self, test_settings, mock_embedding_generator, mock_vector_store):
        """Test that OpenRouter configuration is correctly applied."""
        chatbot = RAGChatBot(
            openrouter_config=test_settings.openrouter,
            embedding_generator=mock_embedding_generator,
            vector_store=mock_vector_store,
        )

        # Verify base URL
        assert str(chatbot.llm.base_url) == "https://openrouter.ai/api/v1"

        # Verify headers include attribution if configured
        headers = chatbot.llm.default_headers
        if test_settings.openrouter_config.app_url:
            assert "HTTP-Referer" in headers
            assert headers["HTTP-Referer"] == test_settings.openrouter_config.app_url
        if test_settings.openrouter_config.app_name:
            assert "X-Title" in headers
            assert headers["X-Title"] == test_settings.openrouter_config.app_name

    @patch('src.chatbot.rag_bot.ChatOpenAI')
    def test_openrouter_api_error_handling(self, mock_chat_openai, test_settings, mock_embedding_generator, mock_vector_store):
        """Test error handling for OpenRouter API failures."""
        # Mock ChatOpenAI to raise an exception
        mock_chat_openai.side_effect = Exception("OpenRouter API rate limit exceeded")

        chatbot = RAGChatBot(
            openrouter_config=test_settings.openrouter_config,
            embedding_generator=mock_embedding_generator,
            vector_store=mock_vector_store,
        )

        # Mock embedding and vector store for successful retrieval
        mock_embedding_generator.generate_single.return_value = np.random.rand(384)
        mock_vector_store.search.return_value = [
            {"content": "Some content", "metadata": {}, "similarity_score": 0.8}
        ]

        # Call ask method - should handle the error gracefully
        result = chatbot.ask("Test question")

        # Assertions for error response
        assert "error occurred" in result["answer"].lower()
        assert result["sources"] == []
        assert result["confidence_score"] == 0.0
        assert result["model_used"] == test_settings.openrouter_config.model

    def test_chat_error_handling(self, test_settings, mock_embedding_generator, mock_vector_store):
        """Test error handling in conversational mode."""
        chatbot = RAGChatBot(
            openrouter_config=test_settings.openrouter_config,
            embedding_generator=mock_embedding_generator,
            vector_store=mock_vector_store,
        )

        # Mock embedding to raise exception
        mock_embedding_generator.generate_single.side_effect = Exception("Embedding generation failed")

        history = []
        result = chatbot.chat("Test question", history)

        # Assertions
        assert "error occurred" in result["answer"].lower()
        assert result["sources"] == []
        assert result["confidence_score"] == 0.0