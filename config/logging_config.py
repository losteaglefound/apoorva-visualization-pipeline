"""Logging configuration with rotating file handlers and console handlers"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


class LoggingConfig:
    """Configure logging with rotating file handlers and console handlers"""
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize logging configuration
        
        Args:
            log_dir: Directory for log files. Defaults to /data/visualization/logs
        """
        if log_dir is None:
            log_dir = os.getenv("LOG_DIR", "/data/visualization/logs")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log levels
        self.file_log_level = logging.DEBUG  # Detailed logs to file
        self.console_log_level = logging.INFO  # Less verbose to console
        
        # Log format
        self.detailed_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.simple_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def _create_rotating_file_handler(
        self,
        log_file: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> logging.handlers.RotatingFileHandler:
        """Create a rotating file handler"""
        log_path = self.log_dir / log_file
        
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        handler.setLevel(self.file_log_level)
        handler.setFormatter(self.detailed_format)
        
        return handler
    
    def _create_console_handler(self) -> logging.StreamHandler:
        """Create a console handler for terminal output"""
        handler = logging.StreamHandler()
        handler.setLevel(self.console_log_level)
        handler.setFormatter(self.simple_format)
        
        return handler
    
    def configure_logger(
        self,
        logger_name: str,
        log_file: str,
        file_max_bytes: int = 10 * 1024 * 1024,
        file_backup_count: int = 5
    ) -> logging.Logger:
        """
        Configure a logger with both file and console handlers
        
        Args:
            logger_name: Name of the logger
            log_file: Name of the log file (will be placed in log_dir)
            file_max_bytes: Maximum size of log file before rotation (default 10MB)
            file_backup_count: Number of backup files to keep (default 5)
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers filter
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Add rotating file handler (detailed logs)
        file_handler = self._create_rotating_file_handler(
            log_file,
            max_bytes=file_max_bytes,
            backup_count=file_backup_count
        )
        logger.addHandler(file_handler)
        
        # Add console handler (simple logs)
        console_handler = self._create_console_handler()
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger


# Global logging configuration instance
_logging_config: Optional[LoggingConfig] = None


def get_logging_config() -> LoggingConfig:
    """Get or create the global logging configuration"""
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    return _logging_config


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger for a service
    
    Args:
        name: Logger name (typically module or service name)
        log_file: Optional log file name. If None, uses name with .log extension
    
    Returns:
        Configured logger
    """
    config = get_logging_config()
    
    if log_file is None:
        log_file = f"{name.replace('.', '_')}.log"
    
    return config.configure_logger(name, log_file)


# Pre-configured loggers for main services
def get_openx_etl_logger() -> logging.Logger:
    """Get logger for Open-X ETL pipeline"""
    return get_logger("roworks.pipelines.openx_etl", "openx_etl.log")


def get_bridgedata_etl_logger() -> logging.Logger:
    """Get logger for Bridgedata ETL pipeline"""
    return get_logger("roworks.pipelines.bridgedata_etl", "bridgedata_etl.log")


def get_analysis_logger() -> logging.Logger:
    """Get logger for analysis layer"""
    return get_logger("roworks.analysis.metrics", "analysis.log")


def get_visualization_logger() -> logging.Logger:
    """Get logger for visualization"""
    return get_logger("roworks.visualization.dashboards", "visualization.log")


def get_clickhouse_logger() -> logging.Logger:
    """Get logger for ClickHouse client"""
    return get_logger("roworks.storage.clickhouse", "clickhouse.log")


def get_s3_logger() -> logging.Logger:
    """Get logger for S3 client"""
    return get_logger("roworks.storage.s3", "s3.log")


def get_main_logger() -> logging.Logger:
    """Get logger for main application"""
    return get_logger("roworks.main", "main.log")

