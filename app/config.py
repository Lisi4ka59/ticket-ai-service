from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "llama.cpp"
    llm_api_key: str | None = None
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "Qwen3-0.6B-Q4_K_M.gguf"
    llm_timeout_seconds: int = Field(default=120, ge=1)
    llm_max_retries: int = Field(default=1, ge=1)
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_max_tokens: int = Field(default=900, ge=128, le=8192)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
