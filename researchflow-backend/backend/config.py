from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://hzh@localhost:5432/researchflow"
    database_url_sync: str = "postgresql://hzh@localhost:5432/researchflow"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Object Storage
    object_storage_provider: str = "cos"  # "cos" or "oss" or "local"
    object_storage_bucket: str = "researchflow"
    object_storage_secret_id: str = ""
    object_storage_secret_key: str = ""
    object_storage_region: str = "ap-shanghai"
    object_storage_cdn_domain: str = ""  # e.g., "cdn.researchflow.xyz" for public URLs

    # GROBID
    grobid_url: str = "http://localhost:8070"

    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""  # Custom base URL for OpenAI-compatible APIs
    openai_model: str = ""     # Custom model name override

    # App
    debug: bool = False
    api_prefix: str = "/api/v1"

    # Paths (relative to project root)
    paper_analysis_dir: str = "../paperAnalysis"
    paper_pdfs_dir: str = "../paperPDFs"
    paper_collection_dir: str = "../paperCollection"
    paper_ideas_dir: str = "../paperIDEAs"
    exports_dir: str = "./exports"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
