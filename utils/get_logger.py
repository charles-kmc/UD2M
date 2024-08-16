import logging
import os

# Set up the logger
def get_loggers(log_dir, log_file='training.log', level=logging.INFO):
    # Create logger
    if log_file.endswith(".log"):
        pass
    else:
        log_file = log_file + ".log"
    
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Full path for the log file
    log_path = os.path.join(log_dir, log_file)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Create file handler which logs to a file
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    
    # Create console handler which logs to the console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Set a simple logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


