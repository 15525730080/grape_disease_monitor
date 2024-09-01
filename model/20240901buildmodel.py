import json
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from pathlib import Path
from collections import defaultdict


class GrapeDiseaseClassifier:
    def __init__(self, data_folder=None, model_path="grape_disease_model"):
        self.data_folder = data_folder
        self.model_path = model_path
        self.le = LabelEncoder()
        self.max_images_per_class = 1000  # 每个类别最大图像数量

    def load_images_from_folder(self):
        images = []
        labels = []
        folder = os.path.abspath(self.data_folder)
        class_images_count = defaultdict(int)
        for img_type in Path(folder).iterdir():
            for file in img_type.iterdir():
                if class_images_count[img_type.name] >= self.max_images_per_class:
                    continue
                file_path = str(file.resolve())
                try:
                    img = cv2.imdecode(np.fromfile(file.resolve(), dtype=np.uint8), -1)
                    if img is not None:
                        img = cv2.resize(img, (224, 224))
                        images.append(img)
                        labels.append(img_type.name)
                        class_images_count[img_type.name] += 1
                    else:
                        print(f"Warning: Unable to read image {file_path}")
                except Exception as e:
                    print(f"Error reading image {file_path}: {e}")
        if len(images) == 0:
            raise ValueError("No images loaded. Please check the dataset path and files.")
        images = np.array(images)
        labels = np.array(labels)
        print(f"Loaded {len(images)} images and {len(labels)} labels., ", set(labels))
        return images, labels

    def preprocess_images(self, images):
        processed_images = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            processed_images.append(img)
        return np.array(processed_images)

    def train_model(self, epochs=25):
        images, labels = self.load_images_from_folder()
        images = self.preprocess_images(images)
        labels_encoded = self.le.fit_transform(labels)
        label_mapping = {int(enc): label for label, enc in zip(labels, labels_encoded)}
        with open("label_mapping.json", "w+") as json_file:
            json_file.write(json.dumps(label_mapping))

        X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

        datagen = ImageDataGenerator(
            rescale=1. / 255,  # 直接在这里进行归一化
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        datagen.fit(X_train)

        model = Sequential([
            Conv2D(64, (3, 3), activation="relu", input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation="relu"),
            GlobalAveragePooling2D(),
            Dropout(0.3),  # 降低Dropout率
            Dense(256, activation="relu"),  # 增加全连接层的神经元数量
            Dense(len(self.le.classes_), activation="softmax")
        ])

        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping]
        )

        model.save(self.model_path + ".keras")  # 保存为 .h5 格式
        print("Model saved to", self.model_path + ".keras")

    def predict(self, image_path):
        model = tf.keras.models.load_model(self.model_path + ".keras")
        with open("label_mapping.json", "r") as json_file:
            label_mapping = json.loads(json_file.read())
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32)  # 确保使用浮点类型
        image_array /= 255.0  # 正确地归一化
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)
        print(f"Predicted class: {label_mapping[str(predicted_class[0])]}")
        probabilities = np.max(predictions, axis=1)
        print(f"Class probabilities: {probabilities[0]}", image_path)


# if __name__ == "__main__":
#     data_folder = r"E:\postgraduatecode\grape_disease_monitor\img\trains"
#     model_path = "grape_disease_model_20240901"
#     classifier = GrapeDiseaseClassifier(data_folder, model_path)
#     classifier.train_model(epochs=50)
#
#     test_image_paths = [
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\溃疡病\6235845bd7561b594fb696e2.jpg",
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\灰霉病\62358460d7561b594fb69a0a.jpg",
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\酸腐病\6235844ed7561b594fb68da2.jpg",
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\黑霉病\6235844dd7561b594fb68ca3.jpg",
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\白粉病\6235844fd7561b594fb68e1f.jpg"
#     ]
#
#     for path in test_image_paths:
#         classifier.predict(path)
