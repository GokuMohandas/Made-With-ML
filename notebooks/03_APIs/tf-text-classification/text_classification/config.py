import os
import sys
sys.path.append(".")
import logging
import logging.config

from text_classification import utils

# Directories
BASE_DIR = os.getcwd()  # project root
APP_DIR = os.path.dirname(__file__)  # app root
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')

# Create dirs
utils.create_dirs(LOGS_DIR)
utils.create_dirs(EMBEDDINGS_DIR)
utils.create_dirs(EXPERIMENTS_DIR)

# Loggers
log_config = utils.load_json(
    filepath=os.path.join(BASE_DIR, 'logging.json'))
logging.config.dictConfig(log_config)
logger = logging.getLogger('logger')
