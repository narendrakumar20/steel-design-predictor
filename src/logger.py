"""
Logging configuration for SteelML.
Provides centralized logging setup with file and console handlers.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from src.config import LoggingConfig


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: Optional[str] = None,
    console_level: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__ of the module)
        log_file: Path to log file (default: from LoggingConfig)
        level: File logging level (default: from LoggingConfig)
        console_level: Console logging level (default: from LoggingConfig)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Application started")
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set logger level to DEBUG to allow all messages through
    logger.setLevel(logging.DEBUG)
    
    # Use config defaults if not specified
    log_file = log_file or LoggingConfig.LOG_FILE
    file_level = getattr(logging, level or LoggingConfig.FILE_LOG_LEVEL)
    console_log_level = getattr(logging, console_level or LoggingConfig.CONSOLE_LOG_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=LoggingConfig.LOG_FORMAT,
        datefmt=LoggingConfig.LOG_DATE_FORMAT
    )
    
    # File handler with rotation
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LoggingConfig.MAX_BYTES,
            backupCount=LoggingConfig.BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def set_log_level(logger: logging.Logger, level: str) -> None:
    """
    Change the log level for all handlers of a logger.
    
    Args:
        logger: Logger instance
        level: New log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    
    Example:
        >>> logger = get_logger(__name__)
        >>> set_log_level(logger, 'DEBUG')
    """
    log_level = getattr(logging, level.upper())
    for handler in logger.handlers:
        handler.setLevel(log_level)


__all__ = ['setup_logger', 'get_logger', 'set_log_level']
