"""Configuration settings for RoWorks data analysis"""

import os
from typing import Optional


class Settings:
    """Application settings"""
    
    # ClickHouse settings
    CLICKHOUSE_HOST: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    CLICKHOUSE_PORT: int = int(os.getenv("CLICKHOUSE_PORT", "9000"))
    CLICKHOUSE_DATABASE: str = os.getenv("CLICKHOUSE_DATABASE", "roworks")
    CLICKHOUSE_USER: str = os.getenv("CLICKHOUSE_USER", "default")
    CLICKHOUSE_PASSWORD: str = os.getenv("CLICKHOUSE_PASSWORD", "")
    
    # S3 settings
    S3_BUCKET: str = os.getenv("S3_BUCKET", "roworks-data")
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
    S3_ACCESS_KEY: Optional[str] = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY: Optional[str] = os.getenv("S3_SECRET_KEY")
    S3_ENDPOINT: Optional[str] = os.getenv("S3_ENDPOINT")
    
    # Data paths
    OPENX_DATA_PATH: str = os.getenv("OPENX_DATA_PATH", "/data/openx")
    BRIDGEDATA_PATH: str = os.getenv("BRIDGEDATA_PATH", "/data/raw")
    
    # Output paths
    OUTPUT_PATH: str = os.getenv("OUTPUT_PATH", "/data/visualization/output")
    
    # Analysis settings
    MIN_SEQUENCE_LENGTH: int = 10  # Minimum steps for training
    MAX_SEQUENCE_LENGTH: int = 10000  # Maximum steps
    
    # Visualization settings
    DASHBOARD_OUTPUT_DIR: str = os.getenv("DASHBOARD_OUTPUT_DIR", "/data/visualization/dashboards")


settings = Settings()

