"""
Logging utilities for SemiCART.

This module provides logging functionality for the SemiCART package.
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str, 
    level: int = logging.INFO, 
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger with the specified name and configuration.
    
    Parameters
    ----------
    name : str
        Name of the logger.
    level : int, optional (default=logging.INFO)
        Logging level.
    log_format : str, optional
        Format string for log messages. If None, a default format is used.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handler if logger doesn't already have handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)
    
    return logger


# Main package logger
logger = get_logger("semicart") 