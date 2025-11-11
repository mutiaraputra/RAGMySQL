import logging
from typing import Dict, List, Optional

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import OpenRouterConfig
from src.embeddings.generator import EmbeddingGenerator
from src.vectorstore.tidb_store import TiDBVectorStore
from src.chatbot.prompts import RAG_TEMPLATE, CONVERSATIONAL_TEMPLATE, FALLBACK_RESPONSE, RAG_SYSTEM_PROMPT


class RAGChatBot:
    """
    RAG ChatBot implementation using LangChain with OpenRouter API.

    This class integrates retrieval from TiDB Vector Store with LLM generation via OpenRouter,
    supporting both single-turn and conversational interactions.
    """

    def __init__(
        self,
        openrouter_config: OpenRouterConfig,
        embedding_generator: EmbeddingGenerator,
        vector_store: TiDBVectorStore,
    ):
        """
        Initialize the RAG ChatBot.

        Args:
            openrouter_config: Configuration for OpenRouter API
            embedding_generator: Instance of EmbeddingGenerator for query embeddings
            vector_store: Instance of TiDBVectorStore for document retrieval
        """
        self.openrouter_config = openrouter_config
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

        # Initialize ChatOpenAI with OpenRouter configuration
        self.llm = ChatOpenAI(
            model=openrouter_config.model,
            api_key=openrouter_config.api_key,
            base_url=str(openrouter_config.base_url),
            default_headers=openrouter_config.headers(),
        )

        # Setup prompt templates
        self.rag_prompt = PromptTemplate(
            template=RAG_TEMPLATE,
            input_variables=["context", "question"],
        )
        self.conversational_prompt = PromptTemplate(
            template=CONVERSATIONAL_TEMPLATE,
            input_variables=["chat_history", "context", "question"],
        )

        self.logger.info(f"Initialized RAGChatBot with model: {openrouter_config.model}")

    def setup_retrieval_chain(self):
        """
        Setup LangChain retrieval chain. Note: Since TiDBVectorStore is custom,
        retrieval is handled manually in ask/chat methods for flexibility.
        """
        # Placeholder for future LangChain integration if needed
        pass

    def ask(self, question: str, top_k: int = 5) -> Dict:
        """
        Answer a single question using RAG.

        Args:
            question: The user's question
            top_k: Number of top documents to retrieve

        Returns:
            Dict with keys: answer, sources, model_used, confidence_score (optional)
        """
        try:
            # Generate embedding for the question
            query_embedding = self.embedding_generator.generate_single(question)
            self.logger.debug(f"Generated query embedding for question: {question[:50]}...")

            # Retrieve relevant documents
            retrieved_docs = self.vector_store.search(query_embedding, top_k)
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents for question")

            # Format context from retrieved documents
            if retrieved_docs:
                context = "\n\n".join([
                    f"Source: {doc['metadata'].get('source_table', 'Basis Data')}\nContent: {doc['content']}"
                    for doc in retrieved_docs
                ])
                confidence_score = sum(doc['similarity_score'] for doc in retrieved_docs) / len(retrieved_docs)
            else:
                context = ""
                confidence_score = 0.0

            # Prepare sources for response
            sources = [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity_score": doc["similarity_score"]
                }
                for doc in retrieved_docs
            ]

            # Generate answer using LLM
            if retrieved_docs:
                prompt = self.rag_prompt.format(context=context, question=question)
                messages = [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                response = self.llm.invoke(input=messages)
                answer = response.content
            else:
                answer = FALLBACK_RESPONSE

            self.logger.info(f"Generated answer using model: {self.openrouter_config.model}")

            return {
                "answer": answer,
                "sources": sources,
                "model_used": self.openrouter_config.model,
                "confidence_score": confidence_score,
            }

        except Exception as e:
            self.logger.error(f"Error in ask method: {e}")
            return {
                "answer": "An error occurred while processing your question. Please try again.",
                "sources": [],
                "model_used": self.openrouter_config.model,
                "confidence_score": 0.0,
            }

    def chat(self, question: str, conversation_history: List[Dict[str, str]]) -> Dict:
        """
        Handle conversational interaction with memory.

        Args:
            question: The current user question
            conversation_history: List of previous turns, each as {"role": "user"/"assistant", "content": str}

        Returns:
            Dict with keys: answer, sources, model_used, confidence_score (optional)
        """
        try:
            # Generate embedding for the question
            query_embedding = self.embedding_generator.generate_single(question)

            # Retrieve relevant documents
            retrieved_docs = self.vector_store.search(query_embedding, top_k=5)

            # Format context
            if retrieved_docs:
                context = "\n\n".join([
                    f"Source: {doc['metadata'].get('source_table', 'Basis Data')}\nContent: {doc['content']}"
                    for doc in retrieved_docs
                ])
                confidence_score = sum(doc['similarity_score'] for doc in retrieved_docs) / len(retrieved_docs)
            else:
                context = ""
                confidence_score = 0.0

            # Format chat history
            chat_history_str = "\n".join([
                f"{turn['role'].capitalize()}: {turn['content']}"
                for turn in conversation_history[-10:]  # Limit to last 10 turns
            ])

            # Prepare sources
            sources = [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity_score": doc["similarity_score"]
                }
                for doc in retrieved_docs
            ]

            # Generate response
            if retrieved_docs:
                prompt = self.conversational_prompt.format(
                    chat_history=chat_history_str,
                    context=context,
                    question=question
                )
                messages = [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                response = self.llm.invoke(input=messages)
                answer = response.content
            else:
                answer = FALLBACK_RESPONSE

            # Update conversation history (in-memory, could be persisted to TiDB if needed)
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})

            self.logger.info(f"Handled conversational turn using model: {self.openrouter_config.model}")

            return {
                "answer": answer,
                "sources": sources,
                "model_used": self.openrouter_config.model,
                "confidence_score": confidence_score,
            }

        except Exception as e:
            self.logger.error(f"Error in chat method: {e}")
            return {
                "answer": "An error occurred during the conversation. Please try again.",
                "sources": [],
                "model_used": self.openrouter_config.model,
                "confidence_score": 0.0,
            }

    def get_sources(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve sources for a question without generating an LLM response.

        Args:
            question: The user's question
            top_k: Number of top documents to retrieve

        Returns:
            List of dicts with content and metadata
        """
        try:
            query_embedding = self.embedding_generator.generate_single(question)
            retrieved_docs = self.vector_store.search(query_embedding, top_k)
            sources = [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity_score": doc["similarity_score"]
                }
                for doc in retrieved_docs
            ]
            self.logger.info(f"Retrieved {len(sources)} sources for question")
            return sources
        except Exception as e:
            self.logger.error(f"Error retrieving sources: {e}")
            return []