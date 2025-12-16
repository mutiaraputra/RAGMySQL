import logging
from typing import Dict, List
import google.generativeai as genai

from config.settings import GeminiConfig
from src.embeddings.generator import EmbeddingGenerator
from src.vectorstore.pinecone_store import PineconeVectorStore  # ✅ Pinecone
from src.chatbot.prompts import RAG_TEMPLATE, FALLBACK_RESPONSE, RAG_SYSTEM_PROMPT


class RAGChatBot:
    """RAG ChatBot with Pinecone vector store and Google Gemini."""

    def __init__(
        self,
        gemini_config: GeminiConfig,  # ✅ Gemini config
        embedding_generator: EmbeddingGenerator,
        vector_store: PineconeVectorStore,  # ✅ Pinecone
    ):
        self.gemini_config = gemini_config
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

        # Configure Gemini
        genai.configure(api_key=gemini_config.api_key)
        
        # Setup generation config
        generation_config = {
            "temperature": gemini_config.temperature,
            "top_p": gemini_config.top_p,
            "top_k": gemini_config.top_k,
            "max_output_tokens": gemini_config.max_output_tokens,
        }
        
        # Setup safety settings if provided
        safety_settings = gemini_config.safety_settings or {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=gemini_config.model,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=RAG_SYSTEM_PROMPT
        )

        self.logger.info(f"Initialized RAGChatBot with Gemini model: {gemini_config.model}")

    def ask(self, question: str, top_k: int = 5) -> Dict:
        """Answer question using RAG with Pinecone retrieval and Gemini."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single(question)

            # Retrieve from Pinecone
            retrieved_docs = self.vector_store.search(query_embedding, top_k)

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
                response = self.model.generate_content(prompt)
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
        """
        Answer question in conversational mode with history.
        
        Args:
            question: User's current question
            history: List of previous messages [{"role": "user"/"assistant", "content": "..."}]
            top_k: Number of documents to retrieve
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single(question)

            # Retrieve from Pinecone
            retrieved_docs = self.vector_store.search(query_embedding, top_k)

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

            # Build chat with context
            if retrieved_docs:
                # Start chat session
                chat = self.model.start_chat(history=[])
                
                # Add conversation history
                for msg in history:
                    if msg["role"] == "user":
                        chat.send_message(msg["content"])
                    # Assistant messages are automatically tracked
