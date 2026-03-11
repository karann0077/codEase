import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq API — get free key at console.groq.com
    groq_api_key: str = ""

  
    github_token: str = ""

    # Groq LLM model (free tier options):
    #   llama-3.3-70b-versatile 
    #   llama-3.1-8b-instant    
    #   mixtral-8x7b-32768      
    chat_model: str = "llama-3.3-70b-versatile"

    # fastembed model — BAAI/bge-small-en-v1.5 → 384 dims, ~50MB download
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # CORS — set to your Vercel frontend URL in production
    # e.g. "https://devpilot.vercel.app" or keep "*" for open access
    cors_origins: str = "*"

    max_file_size_mb: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
