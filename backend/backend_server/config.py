from queue import Queue

from backend.backend_server.log import log as logger
logger.info("loading config start")
NATS_URL = "nats://localhost:4222"
BASE_MESSAGE_QUEUE = Queue()
logger.info("loading config end")
JWT_KEY = "fbz_grape_disease_monitor"