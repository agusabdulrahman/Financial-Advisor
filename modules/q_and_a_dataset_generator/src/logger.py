import logging
from typing import Optional

def get_console_logger(name: Optional[str] = 'tutorial') -> logging.Logger:
    
    # Create logger if it doesn't exist
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.Debug)
        
        # Create conlose hander with formatting
        console_handler = logging.SreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(formatter)
        
        # Add console handler to the logger
        logger.addHandler(console_handler)
        
    return logger    