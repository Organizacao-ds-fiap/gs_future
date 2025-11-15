import logging

def get_logger() -> logging.Logger:
    """Initializes and returns a logger.
    This function sets up a logger with the INFO logging level.

    Returns:
        logging.Logger: A logger instance configured to log at the INFO level.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    return logger