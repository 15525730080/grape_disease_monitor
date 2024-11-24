import time
from io import BytesIO
from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image
import json

__all__ = ["ensemble_predict"]

from predict.mobilenet_v4 import GrapeDiseaseClassifier as MobilenetV4
from predict.mobilenet_v4_SE_FPN import GrapeDiseaseClassifier as MobilenetV4SEFPN
from predict.mobilenet_v4_SE_BiFPN import GrapeDiseaseClassifier as MobilenetV4SEBiFPN
from concurrent.futures import ThreadPoolExecutor, wait


class EnsembleGrapeDiseaseClassifier:

    def __init__(self, model_paths, models):
        self.model_paths = model_paths
        self.models = [tf.keras.models.load_model(path) for path in model_paths]
        self.executor = ThreadPoolExecutor()
        # 假设label_mapping是相同的
        with open(Path(__file__).parent.joinpath("label_mapping.json").resolve(), "r", encoding="utf-8") as json_file:
            self.label_mapping = json.loads(json_file.read())
        self.models = models

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32)
        image_array /= 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def resize_predictions(self, predictions, num_classes):
        resized_preds = []
        for pred in predictions:
            if pred.shape[1] < num_classes:
                # Padding
                pad_width = num_classes - pred.shape[1]
                padded_pred = np.pad(pred, ((0, 0), (0, pad_width)), mode='constant')
            elif pred.shape[1] > num_classes:
                # Trimming
                padded_pred = pred[:, :num_classes]
            else:
                padded_pred = pred
            resized_preds.append(padded_pred)
        return resized_preds

    def predict(self, image_path) -> tuple[str, float]:
        start_predict = time.time()
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32)
        image_array /= 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Collect predictions from all models
        all_predictions = []
        num_classes = None
        # 弱一些的模型需要降低权重
        futures_old = [self.executor.submit(tf.keras.models.load_model(model_path).predict, image_array) for model_path
                       in
                       self.model_paths]
        # 强一点的模型，标准权重
        futures_new = [self.executor.submit(model.predict, image_path) for model in self.models]
        done, undone = wait(futures_old)
        all_predictions = [future.result() for future in done]
        num_classes = max([i.shape[1] for i in all_predictions])
        # Resize predictions to have the same shape
        all_predictions = self.resize_predictions(all_predictions, num_classes)
        # Average predictions across all models
        avg_predictions = np.mean(all_predictions, axis=0)
        # Get the predicted class and probabilities
        predicted_class = np.argmax(avg_predictions, axis=1)
        predicted_class = self.label_mapping[str(predicted_class[0])]
        probabilities = np.max(avg_predictions, axis=1)
        probabilities = probabilities[0]
        new_done, new_undone = wait(futures_new)
        new_all_predictions = [future.result() for future in new_done]
        most_accurate = max(new_all_predictions, key=lambda x: x[1])
        if most_accurate[-1] > (probabilities * 0.5):
            probabilities = most_accurate[-1]
            predicted_class = most_accurate[0]
            print("use new Model")
        print(f"Predicted class: {predicted_class}")
        print(f"Class probabilities: {probabilities * 100} %")
        end_predict = time.time()
        print("预测耗时 {0} 秒".format(end_predict - start_predict))
        return predicted_class, round(float(probabilities * 100), 4)


ensemble_classifier = EnsembleGrapeDiseaseClassifier(
    [
        Path(__file__).parent.joinpath("cnn_v1.keras").resolve(),
        Path(__file__).parent.joinpath("cnn_v2.keras").resolve(),
        # Path(__file__).parent.joinpath("mobilenet_v2.keras").resolve()
    ],
    [
        # mobilenet_v4 mobilenet_v4_SE_FPN mobilenet_v4_SE_BiFPN
        MobilenetV4("", Path(__file__).parent.joinpath("mobilenet_v4.pth").resolve()),
        MobilenetV4SEFPN("", Path(__file__).parent.joinpath("mobilenet_v4_SE_FPN.pth").resolve()),
        MobilenetV4SEBiFPN("", Path(__file__).parent.joinpath("mobilenet_v4_SE_BiFPN.pth").resolve())

    ]
)


def ensemble_predict(img_path: BytesIO | str) -> tuple[str, float]:
    # 使用模型集成进行预测
    return ensemble_classifier.predict(img_path)


#
if __name__ == '__main__':
    test_image_paths = [
        r"E:\postgraduatecode\grape_disease_monitor\img\trains\溃疡病\6235845bd7561b594fb696e2.jpg",
        r"E:\postgraduatecode\grape_disease_monitor\img\trains\灰霉病\62358460d7561b594fb69a0a.jpg",
        r"E:\postgraduatecode\grape_disease_monitor\img\trains\酸腐病\6235844ed7561b594fb68da2.jpg",
        r"E:\postgraduatecode\grape_disease_monitor\img\trains\黑霉病\6235844dd7561b594fb68ca3.jpg",
        r"E:\postgraduatecode\grape_disease_monitor\img\trains\白粉病\6235844fd7561b594fb68e1f.jpg"
    ]

    for path in test_image_paths:
        print(ensemble_predict(path), path)
