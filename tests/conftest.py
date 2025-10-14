import pytest
from unittest.mock import Mock, patch
import numpy as np
from config.settings import Settings

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")

@pytest.fixture
def mock_mysql_connection():
    """Fixture to mock MySQL database connection."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [('table1',), ('table2',)]
    mock_cursor.fetchone.return_value = (100,)
    with patch('mysql.connector.connect', return_value=mock_conn):
        yield mock_conn

@pytest.fixture
def mock_tidb_connection():
    """Fixture to mock TiDB database connection."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    # Patch the TiDB client connect used in the TiDBVectorStore implementation
    with patch('src.vectorstore.tidb_store.TiDBClient.connect', return_value=mock_conn):
        yield mock_conn

@pytest.fixture
def mock_openrouter_client():
    """Fixture to mock OpenRouter API calls using mock ChatOpenAI."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = "Mocked response from OpenRouter"
    mock_client.invoke.return_value = mock_response
    # When ChatOpenAI(...) is called, ensure the returned mock has attributes set from kwargs
    def _factory(*args, **kwargs):
        for k, v in kwargs.items():
            try:
                setattr(mock_client, k, v)
            except Exception:
                pass
        return mock_client

    with patch('langchain_openai.ChatOpenAI', side_effect=_factory):
        yield mock_client


@pytest.fixture
def mock_embedding_generator():
    """Provide a mock embedding generator with expected methods."""
    mock_gen = Mock()
    mock_gen.generate_single = Mock()
    mock_gen.generate_embeddings = Mock()
    mock_gen.get_dimension = Mock(return_value=384)
    return mock_gen


@pytest.fixture
def mock_vector_store():
    """Provide a mock vector store with search and add_documents methods."""
    mock_vs = Mock()
    mock_vs.search = Mock()
    mock_vs.add_documents = Mock()
    mock_vs.create_table = Mock()
    mock_vs.create_vector_index = Mock()
    mock_vs.get_document_count = Mock(return_value=0)
    return mock_vs

@pytest.fixture
def sample_documents():
    """Fixture for sample test documents."""
    return [
        {
            'id': 1,
            'content': 'This is a sample document about machine learning.',
            'metadata': {'source_table': 'articles', 'author': 'John Doe'}
        },
        {
            'id': 2,
            'content': 'Another document discussing artificial intelligence and its applications.',
            'metadata': {'source_table': 'articles', 'author': 'Jane Smith'}
        },
        {
            'id': 3,
            'content': 'Information on natural language processing techniques.',
            'metadata': {'source_table': 'articles', 'author': 'Bob Johnson'}
        }
    ]

@pytest.fixture
def sample_embeddings():
    """Fixture for mock embedding vectors."""
    # Assuming 384 dimensions as per default config
    return np.random.rand(3, 384).astype(np.float32)

@pytest.fixture
def test_settings(monkeypatch):
    """Fixture to override Settings with test values."""
    # Set test environment variables
    monkeypatch.setenv('MYSQL__HOST', 'localhost')
    monkeypatch.setenv('MYSQL__PORT', '3306')
    monkeypatch.setenv('MYSQL__USER', 'test_user')
    monkeypatch.setenv('MYSQL__PASSWORD', 'test_pass')
    monkeypatch.setenv('MYSQL__DATABASE', 'test_db')
    
    monkeypatch.setenv('TIDB__HOST', 'test.tidb.cloud')
    monkeypatch.setenv('TIDB__PORT', '4000')
    monkeypatch.setenv('TIDB__USER', 'test_tidb_user')
    monkeypatch.setenv('TIDB__PASSWORD', 'test_tidb_pass')
    monkeypatch.setenv('TIDB__DATABASE', 'test_tidb_db')
    monkeypatch.setenv('TIDB__USE_TLS', 'true')
    
    monkeypatch.setenv('OPENROUTER__API_KEY', 'dummy_api_key')
    monkeypatch.setenv('OPENROUTER__MODEL', 'openai/gpt-4o-mini')
    monkeypatch.setenv('OPENROUTER__BASE_URL', 'https://openrouter.ai/api/v1')
    
    monkeypatch.setenv('APP__EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    monkeypatch.setenv('APP__EMBEDDING_DIMENSION', '384')
    monkeypatch.setenv('APP__CHUNK_SIZE', '500')
    monkeypatch.setenv('APP__CHUNK_OVERLAP', '50')
    monkeypatch.setenv('APP__BATCH_SIZE', '100')
    monkeypatch.setenv('APP__VECTOR_TABLE_NAME', 'test_knowledge_base')
    monkeypatch.setenv('APP__VECTOR_INDEX_TYPE', 'HNSW')
    monkeypatch.setenv('APP__DISTANCE_METRIC', 'cosine')
    
    # Create and return Settings instance
    settings = Settings()
    yield settings