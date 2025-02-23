import builtins
import os
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.joinpath("model").resolve()))
# 定义 MobileNetV4 模型（确保有 mobilenet_v4.py 文件）
try:
    from .model.src_mobilenet_v4_SE_FPN import MobileNetV4Pro
except:
    from model.src_mobilenet_v4_SE_FPN import MobileNetV4Pro


def get_model_memory_size(model):
    """
    计算模型所占内存大小
    :param model: PyTorch 模型
    :return: 模型所占内存大小（字节）
    """
    total_memory = 0
    for param in model.parameters():
        # 计算每个参数的元素数量
        num_elements = param.numel()
        # 根据数据类型获取每个元素的字节数
        if param.dtype == torch.float32:
            bytes_per_element = 4
        elif param.dtype == torch.float64:
            bytes_per_element = 8
        elif param.dtype == torch.int32:
            bytes_per_element = 4
        elif param.dtype == torch.int64:
            bytes_per_element = 8
        else:
            # 对于其他不常见的数据类型，这里简单假设每个元素 4 字节
            bytes_per_element = 4
        # 计算该参数所占内存大小
        param_memory = num_elements * bytes_per_element
        total_memory += param_memory
    return total_memory


class GrapeDiseaseDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.le = LabelEncoder()
        self.labels_encoded = self.le.fit_transform(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels_encoded[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class GrapeDiseaseClassifier:
    def __init__(self, data_folder=None, model_path="mobilenetv4_SE_FPN.pth"):
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
        print(f"Loaded {len(images)} images and {len(labels)} labels.")
        return images, labels

    def preprocess_images(self, images):
        processed_images = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            processed_images.append(img)
        return np.array(processed_images)

    def train_model(self, epochs=50, batch_size=32, learning_rate=1e-4, patience=5):
        images, labels = self.load_images_from_folder()
        images = self.preprocess_images(images)
        labels_encoded = self.le.fit_transform(labels)
        # 保存标签编码映射
        label_mapping = {idx: label for idx, label in enumerate(self.le.classes_)}
        with open(str(Path(__file__).parent.joinpath("label_mapping.json").resolve()), "w",
                  encoding="utf-8") as json_file:
            json.dump(label_mapping, json_file, ensure_ascii=False)

        X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

        # 定义数据增强和预处理
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 均值
                                 [0.229, 0.224, 0.225])  # ImageNet 标准差
        ])

        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 均值
                                 [0.229, 0.224, 0.225])  # ImageNet 标准差
        ])

        # 创建数据集和数据加载器
        train_dataset = GrapeDiseaseDataset(X_train, y_train, transform=train_transform)
        val_dataset = GrapeDiseaseDataset(X_val, y_val, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 定义模型
        num_classes = len(self.le.classes_)
        model = MobileNetV4Pro("MobileNetV4ConvSmall")
        # 修改最后的分类层
        model.fc = nn.Linear(1280, num_classes)
        # 将模型移动到 GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1,
                                                               verbose=True)

        # 早停机制参数
        early_stopping_patience = patience
        epochs_no_improve = 0
        best_val_loss = float('inf')

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # 训练模型
        best_val_acc = 0.0
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0

            # 遍历训练数据
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset)

            # 验证模型
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

            val_loss = val_loss / len(val_dataset)
            val_acc = val_corrects.double() / len(val_dataset)

            scheduler.step(val_loss)

            # 打印更详细的日志信息
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"- Learning rate: {current_lr:.6f}")
            print(f"- Training loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.4f}")
            print(f"- Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.model_path)
                print(f"** Best model saved with accuracy: {best_val_acc:.4f}")
            else:
                print(f"Validation accuracy did not improve from {best_val_acc:.4f}")
            # 早停机制判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"EarlyStopping counter: {epochs_no_improve} out of {early_stopping_patience}")
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered")
                    break

    def predict(self, image_path):
        # 加载模型和标签映射
        with open(str(Path(__file__).parent.joinpath("label_mapping.json").resolve()), "r",
                  encoding="utf-8") as json_file:
            label_mapping = json.load(json_file)
        num_classes = len(label_mapping)
        model = MobileNetV4Pro("MobileNetV4ConvSmall")
        model.fc = nn.Linear(1280, num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model = model.to(device)
        model.eval()

        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f"Total number of parameters: {total_params}")
        # print(f"Number of trainable parameters: {trainable_params}")

        # 计算模型所占内存大小
        model_memory = get_model_memory_size(model)
        # print(f"Model memory usage: {model_memory} bytes ({model_memory / (1024 * 1024):.2f} MB)")

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 均值
                                 [0.229, 0.224, 0.225])  # ImageNet 标准差
        ])

        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)  # 增加批次维度

        image = image.to(device)
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probability = probabilities[0][preds[0]].item()

        predicted_class = label_mapping[str(preds.item())]
        # print(f"Predicted class: {predicted_class}")
        # print(f"Class probability: {probability:.4f}, Image path: {image_path}")
        return predicted_class, round(probability, 4)


# if __name__ == "__main__":
#     data_folder = r"E:\postgraduatecode\grape_disease_monitor\img\trains"
#     model_path = "mobilenet_v4_SE_FPN.pth"
#
#     classifier = GrapeDiseaseClassifier(data_folder, model_path)
#     #     classifier.train_model(epochs=50, batch_size=32, learning_rate=1e-4, patience=5)
#
#     # 测试预测
#     test_image_paths = [
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\溃疡病\6235845bd7561b594fb696e2.jpg",
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\灰霉病\62358460d7561b594fb69a0a.jpg",
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\酸腐病\6235844ed7561b594fb68da2.jpg",
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\黑霉病\6235844dd7561b594fb68ca3.jpg",
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\白粉病\6235844fd7561b594fb68e1f.jpg",
#         r"E:\postgraduatecode\grape_disease_monitor\img\trains\正常\VCG41N87720898.jpg"
#
#     ]
#
#     for path in test_image_paths:
#         classifier.predict(path)
