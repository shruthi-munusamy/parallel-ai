"""Configuration module for Neural Claim Verification System"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    # API Keys
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
    
    # Model settings
    USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Model names
    STANCE_MODEL = "microsoft/deberta-v3-base"
    EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    SPACY_MODEL = "en_core_web_sm"

config = Config()
