from functools import lru_cache
from typing import Dict, Optional, List

from pydantic import BaseModel, Field, HttpUrl, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JSONEndpointConfig(BaseModel):
    """Configuration for JSON API endpoint as data source."""
    url: HttpUrl = Field(..., description="URL endpoint that returns JSON data")
    method: str = Field("GET", description="HTTP method (GET or POST)")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers for API request")
    auth_token: Optional[str] = Field(None, description="Bearer token for authentication")
    timeout: int = Field(30, description="Request timeout in seconds")
    
    # JSON field mappings
    data_path: str = Field("data", description="JSON path to data array (e.g., 'data' or 'results.items')")
    id_field: str = Field("id", description="Field name for document ID")
    content_fields: List[str] = Field(default_factory=list, description="List of fields to concatenate as content")
    metadata_fields: Optional[List[str]] = Field(None, description="Fields to include in metadata (None = all fields)")

    @field_validator('url', mode='before')
    @classmethod
    def validate_url(cls, v):
        if isinstance(v, str):
            return HttpUrl(v)
        return v


class PineconeConfig(BaseModel):
    """Configuration for Pinecone vector database."""
    api_key: str = Field(..., description="Pinecone API key")
    index_url: HttpUrl = Field(..., description="Pinecone index URL")
    namespace: Optional[str] = Field(None, description="Pinecone namespace")
    batch_size: int = Field(100, description="Batch size for upsert operations")

    @field_validator('index_url', mode='before')
    @classmethod
    def validate_index_url(cls, v):
        if isinstance(v, str):
            return HttpUrl(v)
        return v


class GeminiConfig(BaseModel):
    """Configuration for Google Gemini API integration."""
    api_key: str = Field(..., description="Google Gemini API key")
    model: str = Field("gemini-2.0-flash", description="Gemini model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for response generation")
    max_output_tokens: int = Field(2048, ge=1, description="Maximum tokens in response")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(40, ge=1, description="Top-k sampling parameter")
    safety_settings: Optional[Dict[str, str]] = Field(None, description="Safety settings")


class AppConfig(BaseModel):
    """Application-specific configuration settings."""
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    embedding_dimension: PositiveInt = Field(384, description="Dimension of embedding vectors")
    chunk_size: PositiveInt = Field(500, description="Size of text chunks in characters")
    chunk_overlap: int = Field(50, ge=0, description="Overlap between text chunks")
    batch_size: PositiveInt = Field(100, description="Batch size for processing")


class Settings(BaseSettings):
    """Main application settings."""
    json_endpoint: JSONEndpointConfig
    pinecone: PineconeConfig  # ✅ Gunakan Pinecone
    gemini: GeminiConfig      # ✅ Gunakan Gemini
    app: AppConfig

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__',
        extra='ignore',
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached function to get settings instance (singleton pattern)."""
    return Settings()


class _SettingsProxy:
    """Proxy object for Settings that defers creation."""
    def __init__(self):
        object.__setattr__(self, '_real', None)

    def __getattr__(self, name):
        real = object.__getattribute__(self, '_real')
        if real is not None:
            return getattr(real, name)
        if name in self.__dict__:
            return self.__dict__[name]
        raise RuntimeError("Settings not initialized. Call get_settings() or provide config.")

    def __setattr__(self, name, value):
        if name == '_real':
            object.__setattr__(self, name, value)
            return
        real = object.__getattribute__(self, '_real')
        if real is not None:
            setattr(real, name, value)
        else:
            self.__dict__[name] = value

    def bind_real(self, real: Settings):
        object.__setattr__(self, '_real', real)
        for k, v in list(self.__dict__.items()):
            if k == '_real':
                continue
            try:
                setattr(real, k, v)
            except Exception:
                pass


settings_proxy = _SettingsProxy()


def get_initialized_settings() -> Settings:
    """Get initialized settings, forcing creation if needed."""
    if settings_proxy._real is None:
        settings_proxy.bind_real(Settings())
    return settings_proxy._real


settings = get_initialized_settings()