+----------------+     +------------------+     +------------------+     +-----------------+
|   MySQL DB     | --> | MySQL Scraper    | --> | Text Chunker      | --> | Ingestion       |
| (Source Data)  |     | (Data Extraction)|     | (Text Splitting)  |     | Pipeline        |
+----------------+     +------------------+     +------------------+     +-----------------+
                                                                                      |
                                                                                      v
+----------------+     +------------------+     +------------------+     +-----------------+
| Embedding      | <-- | Ingestion       |     | TiDB Vector      | <-- | Ingestion       |
| Generator      |     | Pipeline        |     | Store            |     | Pipeline        |
| (Local Models) |     | (Orchestration) |     | (HNSW Index)     |     | (Orchestration) |
+----------------+     +------------------+     +------------------+     +-----------------+
                                                                                      |
                                                                                      v
+----------------+     +------------------+     +------------------+     +-----------------+
| RAG ChatBot    | <-- | TiDB Vector      |     | OpenRouter API   | <-- | RAG ChatBot     |
| (LangChain)    |     | Store            |     | (LLM Inference)  |     | (Retrieval)     |
+----------------+     +------------------+     +------------------+     +-----------------+
```

For a more detailed Mermaid diagram, see the project's README.md or visualize at: https://mermaid.live/

## Component Explanations

### MySQL Scraper
**Responsibilities:**
- Establishes connection to MySQL databases using mysql-connector-python
- Extracts data from specified tables with configurable text columns
- Implements batching and pagination for memory-efficient processing of large datasets
- Handles schema inspection and data type validation
- Provides context manager support for proper resource cleanup
- Returns structured data with IDs, concatenated content, and metadata

### Text Chunker
**Responsibilities:**
- Splits long text documents into manageable chunks with configurable size and overlap
- Preserves document metadata and adds chunk-specific information (index, total chunks)
- Supports character-based splitting with context overlap to maintain semantic coherence
- Handles edge cases like empty texts, special characters, and very short documents
- Enables efficient embedding generation by breaking down large texts

### Embedding Generator
**Responsibilities:**
- Loads and manages SentenceTransformer models locally (default: all-MiniLM-L6-v2)
- Converts text chunks to dense vector embeddings with configurable dimensions
- Implements batching for efficient processing of multiple texts
- Normalizes embeddings for cosine similarity calculations
- Provides caching mechanisms to avoid recomputation of identical texts
- Returns numpy arrays with proper dimensionality for vector storage

### TiDB Vector Store
**Responsibilities:**
- Manages connection to TiDB databases using pytidb SDK
- Creates and maintains vector tables with proper schema (ID, content, embedding vector, metadata)
- Implements HNSW indexing for fast similarity search
- Handles bulk insertion with upsert logic for duplicate management
- Performs vector similarity search with optional metadata filtering
- Provides statistics and cleanup operations (delete by source, count documents)

### RAG ChatBot
**Responsibilities:**
- Orchestrates the retrieval-augmented generation process using LangChain
- Generates question embeddings and retrieves relevant documents from TiDB
- Formats retrieved context and manages conversation history
- Interfaces with OpenRouter API via ChatOpenAI for LLM inference
- Provides source attribution and confidence scoring
- Supports both single-turn and conversational modes with memory management

### Ingestion Pipeline
**Responsibilities:**
- Orchestrates the end-to-end data ingestion process across all components
- Coordinates scraping, chunking, embedding generation, and vector storage
- Implements progress tracking and error handling with retry logic
- Provides validation checks before pipeline execution
- Supports incremental updates and checkpoint/resume mechanisms
- Returns comprehensive statistics and audit trails

## Data Flow Diagram

```
User Query
    |
    v
RAG ChatBot (LangChain)
    |
    +--> Generate Query Embedding (Embedding Generator)
    |       |
    |       v
    |   Similarity Search (TiDB Vector Store)
    |       |
    |       v
    +--> Retrieved Documents (Context)
    |
    v
LLM Inference (OpenRouter API via ChatOpenAI)
    |
    v
Response with Sources
    ^
    |
MySQL Data Ingestion Flow:
    |
MySQL Tables --> MySQL Scraper --> Text Chunker --> Embedding Generator --> TiDB Vector Store
```

## Technology Choices and Rationale

### pytidb for Vector Storage
- **Choice:** pytidb SDK over tidb-vector for simplicity and Python-native integration
- **Rationale:** Provides high-level abstractions for table creation, vector fields, and indexing while maintaining control over schema design. Easier to integrate with existing Python ecosystem compared to lower-level tidb-vector library.

### Sentence-Transformers for Embeddings
- **Choice:** Local sentence-transformers models (all-MiniLM-L6-v2)
- **Rationale:** Cost-effective with no API dependencies, runs entirely offline, provides good quality embeddings for general-purpose text. Avoids rate limits and costs associated with cloud embedding services while ensuring data privacy.

### OpenRouter for LLM Inference
- **Choice:** OpenRouter API with single interface to multiple LLM providers
- **Rationale:** Provides access to GPT-4, Claude, Llama, and other models through one API key. Competitive pricing, automatic fallbacks, and model routing capabilities. Compatible with OpenAI SDK for seamless LangChain integration.

### LangChain for RAG Orchestration
- **Choice:** LangChain framework with ChatOpenAI integration
- **Rationale:** Rich ecosystem for RAG implementations, excellent OpenRouter compatibility via ChatOpenAI class, built-in prompt management, and conversation memory support. Simplifies complex retrieval-augmented workflows.

## OpenRouter Integration Details

### ChatOpenAI Configuration
The system configures ChatOpenAI with the following parameters:
- `model`: Configurable via environment (e.g., "anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini")
- `api_key`: OpenRouter API key from configuration
- `base_url`: "https://openrouter.ai/api/v1"
- `default_headers`: Includes attribution headers for analytics

### Attribution Headers
OpenRouter supports optional attribution headers for analytics:
- `HTTP-Referer`: Set to APP_URL if configured
- `X-Title`: Set to APP_NAME if configured
These headers help OpenRouter track usage and provide better support.

### Model Selection Strategy
- **Best Quality:** openai/gpt-4o, anthropic/claude-3.5-sonnet (for complex reasoning)
- **Balanced:** openai/gpt-4o-mini, anthropic/claude-3-haiku (good quality at lower cost)
- **Cost-Effective:** meta-llama/llama-3.1-70b-instruct, google/gemini-pro
- **Fallback:** Automatic routing to available models if primary model is unavailable

### Cost Optimization Tips
- Use smaller models (gpt-4o-mini, claude-3-haiku) for simple queries
- Implement caching for frequent queries
- Monitor token usage and set appropriate context limits
- Leverage OpenRouter's pricing transparency to choose optimal models per use case

## Database Schema Design

### TiDB Vector Table Schema
```sql
CREATE TABLE knowledge_base (
    id VARCHAR(255) PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,  -- Dimension matches embedding model
    metadata JSON,
    source_table VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);