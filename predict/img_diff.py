import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class ImageComparator:
    @staticmethod
    def compare_images(image1_path, image2_path, similarity_threshold=0.8):
        """
        比较两张图片的相似度，判断是否达到给定的相似度阈值。

        :param image1_path: 第一张图片的路径
        :param image2_path: 第二张图片的路径
        :param similarity_threshold: 相似度阈值 (默认值为 0.8)
        :return: bool - 是否相似, float - 实际相似度得分
        """

        def cv2_imread(file_path):
            # 读取支持中文路径的图片
            stream = open(file_path, "rb")
            bytes_data = bytearray(stream.read())
            np_array = np.asarray(bytes_data, dtype=np.uint8)
            return cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

        # 加载图片
        img1 = cv2_imread(image1_path)
        img2 = cv2_imread(image2_path)

        # 检查图片是否成功加载
        if img1 is None or img2 is None:
            raise ValueError("One or both images failed to load. Please check the file paths.")

        # 确保图片大小一致
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # 计算 SSIM 相似度
        score, _ = ssim(img1, img2, full=True)

        # 判断是否达到相似度阈值
        return score >= similarity_threshold, score

#
# # 示例调用
# if __name__ == "__main__":
#     image1 = r"E:\postgraduatecode\grape_disease_monitor\img\trains\溃疡病\6235845bd7561b594fb696e4.jpg"
#     image2 = r"E:\postgraduatecode\grape_disease_monitor\img\trains\溃疡病\6235845bd7561b594fb696e7.jpg"
#
#     # 调用静态方法比较图片相似度
#     is_similar, similarity_score = ImageComparator.compare_images(image1, image2, similarity_threshold=0.8)
#
#     if is_similar:
#         print(f"The images are similar with a similarity score of {similarity_score:.2f}.")
#     else:
#         print(f"The images are not similar. Similarity score: {similarity_score:.2f}.")
