import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import get_initialized_settings
from src.chatbot.rag_bot import RAGChatBot
from src.embeddings.generator import EmbeddingGenerator
from src.vectorstore.pinecone_store import PineconeVectorStore  # ✅ Benar
from src.pipeline.ingestion import IngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG ChatBot API",
    description="Retrieval Augmented Generation ChatBot with Google Gemini",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    conversation_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    model_used: str
    confidence_score: float

class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]

class IngestionRequest(BaseModel):
    dry_run: Optional[bool] = False

class IngestionResponse(BaseModel):
    status: str
    statistics: Dict

# Global instances (initialized on startup)
settings = None
chatbot = None
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global settings, chatbot, pipeline
    
    try:
        logger.info("Initializing application components...")
        
        settings = get_initialized_settings()
        embedding_generator = EmbeddingGenerator()
        
        # Initialize vector store - Pinecone
        logger.info("Connecting to Pinecone...")
        vector_store = PineconeVectorStore(
            settings.pinecone,
            settings.app.embedding_dimension
        )
        
        # Initialize chatbot - Gemini
        logger.info("Initializing Gemini chatbot...")
        chatbot = RAGChatBot(
            gemini_config=settings.gemini,  # ✅ Gunakan Gemini
            embedding_generator=embedding_generator,
            vector_store=vector_store,
        )
        
        # Initialize pipeline
        logger.info("Initializing ingestion pipeline...")
        pipeline = IngestionPipeline()
        
        logger.info("Application initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "chatbot": "initialized" if chatbot else "not initialized",
            "pipeline": "initialized" if pipeline else "not initialized"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
    if not chatbot or not pipeline:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "chatbot": "healthy",
            "pipeline": "healthy",
            "vector_store": "connected",
            "gemini": "configured"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint - ask questions with RAG."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        if request.conversation_history:
            # Use conversational mode
            result = chatbot.chat(
                request.question,
                request.conversation_history,
                top_k=request.top_k
            )
        else:
            # Use single-turn mode
            result = chatbot.ask(
                request.question,
                top_k=request.top_k
            )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_data(request: IngestionRequest):
    """Ingest data from JSON endpoint to Pinecone."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        if request.dry_run:
            # Validate only
            is_valid = pipeline.validate_pipeline()
            return {
                "status": "validated" if is_valid else "invalid",
                "message": "Pipeline validation successful" if is_valid else "Validation failed",
                "endpoint": str(settings.json_endpoint.url)
            }
        
        # Run full pipeline (tanpa 'with' statement)
        stats = pipeline.run_full_pipeline()
        
        return {
            "status": "completed",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources")
async def get_sources(
    question: str = Query(..., description="Question to retrieve sources for"),
    top_k: int = Query(5, description="Number of sources to retrieve")
):
    """Get relevant sources without generating an answer."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        sources = chatbot.get_sources(question, top_k=top_k)
        return {"sources": sources}
        
    except Exception as e:
        logger.error(f"Sources retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)