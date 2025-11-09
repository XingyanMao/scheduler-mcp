"""
Enhanced configuration for MCP Scheduler with multi-AI model support.
"""
import os
import json
from typing import Optional, Dict, Any
from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # 回退到旧版本pydantic的BaseSettings
    from pydantic import BaseSettings


class Config(BaseSettings):
    """Configuration for MCP Scheduler."""
    
    # Server configuration
    server_name: str = Field(default="mcp-scheduler")
    server_version: str = Field(default="1.0.0")
    server_address: str = Field(default="localhost")
    server_port: int = Field(default=8080)
    transport: str = Field(default="stdio")  # "stdio" or "sse"
    
    # Database configuration
    db_path: str = Field(default="scheduler.db")
    
    # Logging configuration
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)
    
    # Scheduler configuration
    check_interval: int = Field(default=5)  # seconds
    execution_timeout: int = Field(default=300)  # seconds
    
    # AI configuration - multiple model support
    ai_provider: str = Field(default="openai")  # openai, azure, anthropic, local
    ai_model: str = Field(default="gpt-3.5-turbo")
    
    # OpenAI configuration
    openai_api_key: Optional[str] = Field(default=None)
    openai_base_url: Optional[str] = Field(default="https://api.openai.com/v1")
    openai_organization: Optional[str] = Field(default=None)
    
    # Azure OpenAI configuration
    azure_openai_api_key: Optional[str] = Field(default=None)
    azure_openai_endpoint: Optional[str] = Field(default=None)
    azure_openai_api_version: Optional[str] = Field(default="2023-12-01-preview")
    
    # Anthropic configuration
    anthropic_api_key: Optional[str] = Field(default=None)
    anthropic_base_url: Optional[str] = Field(default="https://api.anthropic.com")
    
    # Local model configuration (Ollama, vLLM, etc.)
    local_model_base_url: Optional[str] = Field(default="http://localhost:11434/v1")
    local_model_api_key: Optional[str] = Field(default="ollama")  # Ollama doesn't require real key
    
    # Additional AI configuration
    ai_max_tokens: int = Field(default=2000)
    ai_temperature: float = Field(default=0.7)
    
    class Config:
        env_prefix = "MCP_SCHEDULER_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = 'utf-8'
        
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            # Customize configuration loading order
            return (
                init_settings,
                env_settings,
                Config.json_config_settings,
                file_secret_settings,
            )
    
    @classmethod
    def json_config_settings(cls, settings: BaseSettings) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_file = os.getenv("MCP_SCHEDULER_CONFIG_FILE")
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI configuration based on selected provider."""
        base_config = {
            "max_tokens": self.ai_max_tokens,
            "temperature": self.ai_temperature,
            "model": self.ai_model
        }
        
        if self.ai_provider == "openai":
            return {
                **base_config,
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "organization": self.openai_organization
            }
        elif self.ai_provider == "azure":
            return {
                **base_config,
                "api_key": self.azure_openai_api_key,
                "base_url": f"{self.azure_openai_endpoint}/openai/deployments/{self.ai_model}",
                "api_version": self.azure_openai_api_version
            }
        elif self.ai_provider == "anthropic":
            return {
                **base_config,
                "api_key": self.anthropic_api_key,
                "base_url": self.anthropic_base_url
            }
        elif self.ai_provider == "local":
            return {
                **base_config,
                "api_key": self.local_model_api_key,
                "base_url": self.local_model_base_url
            }
        else:
            raise ValueError(f"Unsupported AI provider: {self.ai_provider}")