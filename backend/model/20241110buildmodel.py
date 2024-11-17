import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np
import onnxruntime as ort
from backend.model.mobilenetv4 import mobilenetv4_conv_small
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GrapeDiseaseModel:
    def __init__(self, data_dir, test_dir, num_classes, batch_size=64, num_epochs=50, patience=10, learning_rate=0.001):
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 加载数据集
        self.dataset = torchvision.datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=self.transform)
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # 初始化模型、损失函数、优化器
        self.model = mobilenetv4_conv_small(num_classes=self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.early_stopping = EarlyStopping(patience=self.patience, delta=0.01)

    def train(self):
        print("\n=== Training Process Started ===\n")
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            all_labels = []
            all_preds = []

            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx + 1}/{len(self.train_loader)}: "
                          f"Loss: {loss.item():.4f}, Avg Loss: {running_loss / (batch_idx + 1):.4f}")

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = accuracy_score(all_labels, all_preds)
            print(f"Epoch {epoch + 1} Summary: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            val_loss = self.validate()
            self.scheduler.step(val_loss)
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        print("\n=== Training Complete ===\n")

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Validation Results - Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, "
              f"Precision: {precision:.4f}, F1: {f1:.4f}")
        return val_loss

    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_preds = []

        print("\n=== Evaluation Started ===")
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                batch_accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
                print(f"Batch {batch_idx + 1}/{len(self.val_loader)}: Batch Accuracy: {batch_accuracy:.4f}")

        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"\nOverall Evaluation - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, "
              f"Precision: {precision:.4f}, F1 Score: {f1:.4f}\n")
        return accuracy, recall, precision, f1

    def save_model_as_onnx(self, save_path):
        self.model.eval()
        dummy_input = torch.randn(1, 3, 128, 128, device=self.device)
        torch.onnx.export(self.model, dummy_input, save_path, input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, opset_version=11)
        print(f"Model successfully saved as ONNX at {save_path}")


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


if __name__ == '__main__':
    data_dir = r"E:\postgraduatecode\grape_disease_monitor\img\trains_new"
    test_dir = r"E:\postgraduatecode\grape_disease_monitor\img\trains"
    num_classes = len(os.listdir(data_dir))
    model = GrapeDiseaseModel(data_dir=data_dir, test_dir=test_dir, num_classes=num_classes)
    model.train()
    model.evaluate()
