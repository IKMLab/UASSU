import logging
import os
from datetime import datetime

dir_path = './sum_dist/logs'
filename = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.log'

def get_logger(log_name, log_level="INFO"):
    if not os.path.exists(f'{dir_path}'):
        os.makedirs(f'{dir_path}')

    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        '\n%(asctime)s %(name)-5s === %(levelname)-5s === %(message)s\n')

    file_handler = logging.FileHandler(
        filename=f'{dir_path}/{filename}', 
        mode='w', 
        encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
