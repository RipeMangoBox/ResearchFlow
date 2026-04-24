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

    # Semantic Scholar
    s2_api_key: str = ""  # Semantic Scholar API key for higher rate limits

    # GROBID
    grobid_url: str = "http://localhost:8070"

    # MCP Auth
    mcp_auth_token: str = ""  # Set in .env for production

    # OpenReview
    openreview_username: str = ""
    openreview_password: str = ""

    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""  # Custom base URL for OpenAI-compatible APIs
    openai_model: str = ""     # Custom model name override

    # VLM max_tokens — Kimi K2.6 需要开大，否则截断
    vlm_max_tokens_heavy: int = 16384   # 公式页扫描 (多页多公式, 大表格)
    vlm_max_tokens_medium: int = 8192   # 图表分类+遗漏恢复
    vlm_max_tokens_light: int = 4096    # 单图描述/单公式 OCR
    vlm_max_tokens_tiny: int = 2048     # acceptance 判断等短回复

    # GitHub
    github_token: str = ""  # Personal access token for 5000 req/hr (vs 10 req/min)

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
