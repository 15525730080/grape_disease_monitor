from io import BytesIO
from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image
import json

__all__ = ["ensemble_predict"]


class EnsembleGrapeDiseaseClassifier:
    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.models = [tf.keras.models.load_model(path) for path in model_paths]

        # 假设label_mapping是相同的
        with open(Path(__file__).parent.joinpath("label_mapping.json").resolve(), "r") as json_file:
            self.label_mapping = json.loads(json_file.read())

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
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32)
        image_array /= 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Collect predictions from all models
        all_predictions = []
        num_classes = None
        for model_path in self.model_paths:
            model = tf.keras.models.load_model(model_path)
            pred = model.predict(image_array)
            print(f"{model_path} Model prediction shape: {pred.shape}")
            all_predictions.append(pred)
            if num_classes is None:
                num_classes = pred.shape[1]
            else:
                num_classes = max(num_classes, pred.shape[1])

        # Resize predictions to have the same shape
        all_predictions = self.resize_predictions(all_predictions, num_classes)

        # Average predictions across all models
        avg_predictions = np.mean(all_predictions, axis=0)

        # Get the predicted class and probabilities
        predicted_class = np.argmax(avg_predictions, axis=1)
        probabilities = np.max(avg_predictions, axis=1)

        print(f"Predicted class: {self.label_mapping[str(predicted_class[0])]}")
        print(f"Class probabilities: {probabilities[0] * 100} %")

        return predicted_class, float(probabilities[0] * 100)


ensemble_classifier = EnsembleGrapeDiseaseClassifier(
    [
        Path(__file__).parent.joinpath("grape_disease_model_20240831.keras").resolve(),
        Path(__file__).parent.joinpath("grape_disease_model_20240901.keras").resolve(),
        Path(__file__).parent.joinpath("grape_disease_model_20240914_rnn.keras").resolve()
    ])


def ensemble_predict(img_path: BytesIO) -> tuple[str, float]:
    # 使用模型集成进行预测
    return ensemble_classifier.predict(img_path)
