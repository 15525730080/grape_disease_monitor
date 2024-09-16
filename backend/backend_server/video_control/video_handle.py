import time
import traceback
from io import BytesIO
from threading import Thread
from backend.backend_server.config import BASE_MESSAGE_QUEUE
from backend.model.ensemble_classifier import ensemble_predict
from backend.backend_server.log import log as logger
__all__ = ["video_image_handler"]


class VideoImageHandler(object):

    def process_image(self, image_data: str):
        logger.info((BASE_MESSAGE_QUEUE, BASE_MESSAGE_QUEUE.qsize()))
        BASE_MESSAGE_QUEUE.put({"author": "fbz", "image_data": image_data})

    def __custom_message(self):
        if not BASE_MESSAGE_QUEUE.empty():
            item_message = BASE_MESSAGE_QUEUE.get()
            image_data = item_message.get("image_data")
            logger.info("custom")
            ensemble_predict(BytesIO(image_data))

    def start_custom_thread(self):
        def func():
            logger.info("start custom")
            while True:
                try:
                    if BASE_MESSAGE_QUEUE.empty():
                        logger.info("BASE_MESSAGE_QUEUE empty")
                        time.sleep(3)
                        continue
                    logger.info("check custom")
                    self.__custom_message()
                except:
                    logger.error(traceback.print_exc())
                    time.sleep(3)
        Thread(target=func).start()


video_image_handler = VideoImageHandler()
