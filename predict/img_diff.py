import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class ImageComparator:
    @staticmethod
    def compare_images(image1_bytes, image2_bytes, similarity_threshold=0.8):
        """
        比较两张图片的相似度，判断是否达到给定的相似度阈值。

        :param image1_bytes: 第一张图片的字节数据
        :param image2_bytes: 第二张图片的字节数据
        :param similarity_threshold: 相似度阈值 (默认值为 0.8)
        :return: bool - 是否相似, float - 实际相似度得分
        """

        def bytes_to_cv2_image(byte_data):
            # 将字节数据转换为 OpenCV 图片格式
            np_array = np.frombuffer(byte_data, dtype=np.uint8)
            return cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

        # 将字节数据转化为 OpenCV 图像
        img1 = bytes_to_cv2_image(image1_bytes)
        img2 = bytes_to_cv2_image(image2_bytes)

        # 检查图片是否成功加载
        if img1 is None or img2 is None:
            raise ValueError("One or both images failed to load from byte data.")

        # 确保图片大小一致
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # 计算 SSIM 相似度
        score, _ = ssim(img1, img2, full=True)

        # 判断是否达到相似度阈值
        return score >= similarity_threshold, score

