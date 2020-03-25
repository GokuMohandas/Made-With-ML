import os
import logging
import logging.config

import utilities as utils

# Directories
BASE_DIR = os.getcwd()  # project root
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')
TENSORBOARD_DIR = os.path.join(BASE_DIR, 'tensorboard')

# Create dirs
utils.create_dirs(LOGS_DIR)
utils.create_dirs(EXPERIMENTS_DIR)
utils.create_dirs(TENSORBOARD_DIR)

# Loggers
log_config = utils.load_json(
    filepath=os.path.join(BASE_DIR, 'logging.json'))
logging.config.dictConfig(log_config)
logger = logging.getLogger('logger')
