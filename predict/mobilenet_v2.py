# 数据加载和拆分部分
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_split_data(data_folder, max_images_per_class=1000, test_size=0.2, validation_split=0.2):
    images = []
    labels = []
    folder = os.path.abspath(data_folder)
    class_images_count = defaultdict(int)

    for img_type in Path(folder).iterdir():
        for file in img_type.iterdir():
            if class_images_count[img_type.name] >= max_images_per_class:
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

    # 标签编码
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # 保存标签映射
    with open("label_mapping.json", "w+") as json_file:
        label_mapping = {int(enc): label for enc, label in enumerate(le.classes_)}
        json.dump(label_mapping, json_file)

    # 数据拆分
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels_encoded, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_split, random_state=42)

    print(f"Data loaded: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples.")
    return X_train, X_val, X_test, y_train, y_val, y_test, le


# 使用新的数据加载方式
class GrapeDiseaseClassifier:
    def __init__(self, data_folder=None, model_path="mobilenet_v2"):
        self.data_folder = data_folder
        self.model_path = model_path

    def preprocess_images(self, images):
        processed_images = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            processed_images.append(img)
        return np.array(processed_images)

    def build_model(self, num_classes):
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_model.trainable = False  # Freeze pretrained model weights

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dense(num_classes, activation="softmax")
        ])

        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    def train_model(self, epochs=50, batch_size=32, learning_rate=1e-4):
        X_train, X_val, X_test, y_train, y_val, y_test, le = load_and_split_data(self.data_folder)
        num_classes = len(le.classes_)

        X_train = self.preprocess_images(X_train)
        X_val = self.preprocess_images(X_val)
        X_test = self.preprocess_images(X_test)

        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        datagen.fit(X_train)

        model = self.build_model(num_classes)

        early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val / 255.0, y_val),
            callbacks=[early_stopping]
        )

        model.save(self.model_path + ".keras")
        print("Model saved to", self.model_path + ".keras")

        test_loss, test_accuracy = model.evaluate(X_test / 255.0, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

