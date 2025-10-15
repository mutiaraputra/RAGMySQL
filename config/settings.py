from functools import lru_cache
from typing import Dict, Optional

from pydantic import BaseModel, Field, HttpUrl, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MySQLConfig(BaseModel):
    """Configuration for MySQL database connection."""
    host: str = Field(..., description="MySQL server host")
    port: int = Field(3306, gt=0, description="MySQL server port")
    user: str = Field(..., description="MySQL username")
    password: str = Field(..., description="MySQL password")
    database: str = Field(..., description="MySQL database name")


class TiDBConfig(BaseModel):
    """Configuration for TiDB database connection."""
    host: str = Field(..., description="TiDB server host")
    port: int = Field(4000, gt=0, description="TiDB server port")
    user: str = Field(..., description="TiDB username")
    password: str = Field(..., description="TiDB password")
    database: str = Field(..., description="TiDB database name")
    use_tls: bool = Field(False, description="Whether to use TLS for TiDB connection")


class OpenRouterConfig(BaseModel):
    """Configuration for OpenRouter API integration.

    OpenRouter provides access to multiple LLM providers (OpenAI, Anthropic, Meta, etc.) through a single API.
    For model selection, refer to https://openrouter.ai/models for available models, pricing, and performance details.
    """
    api_key: str = Field(..., description="OpenRouter API key")
    model: str = Field("anthropic/claude-3.5-sonnet", description="OpenRouter model name (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-3-haiku')")
    base_url: HttpUrl = Field("https://openrouter.ai/api/v1", description="OpenRouter API base URL")
    app_name: Optional[str] = Field(None, description="Application name for attribution headers (X-Title)")
    app_url: Optional[str] = Field(None, description="Application URL for attribution headers (HTTP-Referer)")

    # Note: env nested delimiter is used to map nested env vars (e.g., OPENROUTER__API_KEY)

    @field_validator('base_url', mode='before')
    @classmethod
    def validate_base_url(cls, v):
        if isinstance(v, str):
            return HttpUrl(v)
        return v

    def headers(self) -> Dict[str, str]:
        """Return attribution headers for OpenRouter API calls (from the OpenRouter config)."""
        headers: Dict[str, str] = {}
        if self.app_url:
            headers["HTTP-Referer"] = str(self.app_url)
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers


class AppConfig(BaseModel):
    """Application-specific configuration settings."""
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Sentence-transformers model for embeddings")
    embedding_dimension: PositiveInt = Field(384, description="Dimension of embedding vectors")
    chunk_size: PositiveInt = Field(500, description="Size of text chunks in characters")
    chunk_overlap: int = Field(50, ge=0, description="Overlap between text chunks in characters")
    batch_size: PositiveInt = Field(100, description="Batch size for processing operations")
    vector_table_name: str = Field("knowledge_base", description="Name of the vector table in TiDB")
    vector_index_type: str = Field("HNSW", description="Type of vector index (e.g., HNSW)")
    distance_metric: str = Field("cosine", description="Distance metric for vector similarity (e.g., cosine, L2)")


class Settings(BaseSettings):
    """Main application settings with nested configurations.

    This class loads configuration from environment variables and .env file.
    It implements a singleton pattern using caching for efficient loading.
    """
    mysql: MySQLConfig
    tidb: TiDBConfig
    openrouter: OpenRouterConfig
    app: AppConfig

    # Use AliasChoices to allow validation by alias which permits flat env names to be used
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__',
        extra='ignore',
    )

    def get_mysql_connection_string(self) -> str:
        """Generate MySQL connection string (DSN)."""
        return f"mysql+mysqlconnector://{self.mysql.user}:{self.mysql.password}@{self.mysql.host}:{self.mysql.port}/{self.mysql.database}"

    def get_tidb_connection_string(self) -> str:
        """Generate TiDB connection string (DSN)."""
        protocol = "mysql" if not self.tidb.use_tls else "mysql+ssl"
        return f"{protocol}://{self.tidb.user}:{self.tidb.password}@{self.tidb.host}:{self.tidb.port}/{self.tidb.database}"

    def get_openrouter_headers(self) -> Dict[str, str]:
        """Generate headers for OpenRouter API requests, including attribution headers if configured."""
        headers = {}
        if self.openrouter.app_url:
            headers["HTTP-Referer"] = str(self.openrouter.app_url)
        if self.openrouter.app_name:
            headers["X-Title"] = self.openrouter.app_name
        return headers


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached function to get settings instance (singleton pattern)."""
    return Settings()


class _SettingsProxy:
    """A proxy object for Settings that defers creation and allows early attribute patching.

    This avoids forcing validation at import-time (which fails when required env vars are missing).
    The proxy stores any attributes set on it until a real Settings instance is created, at which
    point subsequent attribute access is forwarded to the real instance.
    """

    def __init__(self):
        # store real Settings instance here once created
        object.__setattr__(self, '_real', None)

    def __getattr__(self, name):
        real = object.__getattribute__(self, '_real')
        if real is not None:
            return getattr(real, name)
        # If attribute set earlier on proxy, return it
        if name in self.__dict__:
            return self.__dict__[name]
        raise RuntimeError("Settings not initialized. Call get_settings() or provide a config via --config.")

    def __setattr__(self, name, value):
        if name == '_real':
            object.__setattr__(self, name, value)
            return
        real = object.__getattribute__(self, '_real')
        if real is not None:
            setattr(real, name, value)
        else:
            # Store on proxy so code that patches attributes (e.g., main.py) works before real settings exist
            self.__dict__[name] = value

    def bind_real(self, real: Settings):
        """Bind a real Settings instance to the proxy and copy any previously set attributes onto it."""
        object.__setattr__(self, '_real', real)
        # copy attributes that were set on proxy into real
        for k, v in list(self.__dict__.items()):
            if k == '_real':
                continue
            try:
                setattr(real, k, v)
            except Exception:
                # ignore attributes that can't be set on the real Settings
                pass


# Export a module-level proxy for settings so imports succeed even when env is incomplete
settings_proxy = _SettingsProxy()

# Function to get actual settings (forces initialization)
def get_initialized_settings() -> Settings:
    """Get initialized settings, forcing creation if needed."""
    if settings_proxy._real is None:
        settings_proxy.bind_real(Settings())
    return settings_proxy._real

# For backward compatibility, keep 'settings' name but make it callable
settings = get_initialized_settings()