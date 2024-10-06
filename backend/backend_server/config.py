from queue import Queue
from backend.backend_server.log import log as logger

logger.info("loading config start")
JWT_KEY = "fbz_grape_disease_monitor"
BASE_MESSAGE_QUEUE = Queue()
logger.info("loading config end")
