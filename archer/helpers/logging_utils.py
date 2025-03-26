"""
Logger utility for the archer system.

This module provides a centralized logging setup for the entire system,
ensuring consistent logging format and behavior across all components.
"""

import logging
import os
import sys
from pathlib import Path
import time
from functools import wraps

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # Log to console
        logging.StreamHandler(sys.stdout),
        # Log to file with timestamp in filename
        logging.FileHandler(
            logs_dir / f"archer_{time.strftime('%Y%m%d-%H%M%S')}.log"
        )
    ]
)

def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name: The name of the logger, typically the module name
        
    Returns:
        Logger: A configured logger instance
    """
    return logging.getLogger(name)

def log_entry_exit(logger):
    """
    Decorator to log function entry and exit, with timing information.
    
    Args:
        logger: The logger to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Entering {func_name}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(f"Exiting {func_name} - took {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Error in {func_name} after {elapsed:.3f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator

def log_call_args(logger, level=logging.DEBUG):
    """
    Decorator to log function arguments.
    
    Args:
        logger: The logger to use
        level: The logging level to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            
            logger.log(level, f"Calling {func.__name__}({signature})")
            return func(*args, **kwargs)
                
        return wrapper
    return decorator 