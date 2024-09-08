import base64
from io import BytesIO

from PIL.Image import Image

from backend.model.ensemble_classifier import ensemble_predict

__all__ = ["video_image_handler"]


class VideoImageHandler(object):

    def process_image(self, image_data: str):
        ensemble_predict(BytesIO(image_data))


video_image_handler = VideoImageHandler()
