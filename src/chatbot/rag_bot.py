import logging
from typing import Dict, List
from google import genai
from google.genai import types

from config.settings import GeminiConfig
from src.embeddings.generator import EmbeddingGenerator
from src.vectorstore.pinecone_store import PineconeVectorStore
from src.chatbot.prompts import RAG_TEMPLATE, FALLBACK_RESPONSE, RAG_SYSTEM_PROMPT


class RAGChatBot:
    """RAG ChatBot with Pinecone vector store and Google Gemini."""

    def __init__(
        self,
        gemini_config: GeminiConfig,
        embedding_generator: EmbeddingGenerator,
        vector_store: PineconeVectorStore,
    ):
        self.gemini_config = gemini_config
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

        # Configure Gemini client (new SDK)
        self.client = genai.Client(api_key=gemini_config.api_key)
        
        # Store config for generation
        self.generation_config = types.GenerateContentConfig(
            temperature=gemini_config.temperature,
            top_p=gemini_config.top_p,
            top_k=gemini_config.top_k,
            max_output_tokens=gemini_config.max_output_tokens,
            system_instruction=RAG_SYSTEM_PROMPT,
        )

        self.logger.info(f"Initialized RAGChatBot with Gemini model: {gemini_config.model}")

    def ask(self, question: str, top_k: int = 5) -> Dict:
        """Answer question using RAG with Pinecone retrieval and Gemini."""
        try:
            # Retrieve from Pinecone using text search (Pinecone auto-embeds)
            retrieved_docs = self.vector_store.search_by_text(question, top_k)

            # Format context
            if retrieved_docs:
                context = "\n\n".join([
                    f"Source {i+1}:\n{doc['content']}"
                    for i, doc in enumerate(retrieved_docs)
                ])
                confidence_score = sum(
                    doc['similarity_score'] for doc in retrieved_docs
                ) / len(retrieved_docs)
            else:
                context = ""
                confidence_score = 0.0

            sources = [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity_score": doc["similarity_score"]
                }
                for doc in retrieved_docs
            ]

            # Generate answer using Gemini
            if retrieved_docs:
                prompt = RAG_TEMPLATE.format(context=context, question=question)
                response = self.client.models.generate_content(
                    model=self.gemini_config.model,
                    contents=prompt,
                    config=self.generation_config,
                )
                answer = response.text
            else:
                answer = FALLBACK_RESPONSE

            return {
                "answer": answer,
                "sources": sources,
                "model_used": self.gemini_config.model,
                "confidence_score": confidence_score,
            }

        except Exception as e:
            self.logger.error(f"Error in ask: {e}")
            return {
                "answer": f"An error occurred: {str(e)}. Please try again.",
                "sources": [],
                "model_used": self.gemini_config.model,
                "confidence_score": 0.0,
            }

    def chat(self, question: str, history: List[Dict[str, str]], top_k: int = 5) -> Dict:
        """Answer question in conversational mode with history."""
        try:
            # Retrieve from Pinecone using text search
            retrieved_docs = self.vector_store.search_by_text(question, top_k)

            # Format context
            if retrieved_docs:
                context = "\n\n".join([
                    f"Source {i+1}:\n{doc['content']}"
                    for i, doc in enumerate(retrieved_docs)
                ])
                confidence_score = sum(
                    doc['similarity_score'] for doc in retrieved_docs
                ) / len(retrieved_docs)
            else:
                context = ""
                confidence_score = 0.0

            sources = [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity_score": doc["similarity_score"]
                }
                for doc in retrieved_docs
            ]

            # Build conversation context
            conversation_context = ""
            if history:
                conversation_context = "\n".join([
                    f"{msg['role'].capitalize()}: {msg['content']}"
                    for msg in history
                ])
                conversation_context += f"\nUser: {question}"
            else:
                conversation_context = question

            # Generate response
            if retrieved_docs:
                prompt = RAG_TEMPLATE.format(
                    context=context, 
                    question=conversation_context
                )
                response = self.client.models.generate_content(
                    model=self.gemini_config.model,
                    contents=prompt,
                    config=self.generation_config,
                )
                answer = response.text
            else:
                answer = FALLBACK_RESPONSE

            return {
                "answer": answer,
                "sources": sources,
                "model_used": self.gemini_config.model,
                "confidence_score": confidence_score,
            }

        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            return {
                "answer": f"An error occurred: {str(e)}. Please try again.",
                "sources": [],
                "model_used": self.gemini_config.model,
                "confidence_score": 0.0,
            }

    def get_sources(self, question: str, top_k: int = 5) -> List[Dict]:
        """Retrieve sources without generating answer."""
        try:
            retrieved_docs = self.vector_store.search_by_text(question, top_k)
            
            return [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity_score": doc["similarity_score"]
                }
                for doc in retrieved_docs
            ]
        except Exception as e:
            self.logger.error(f"Error getting sources: {e}")
            return []