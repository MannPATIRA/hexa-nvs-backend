from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    supabase_url: str
    supabase_service_key: str
    env: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()