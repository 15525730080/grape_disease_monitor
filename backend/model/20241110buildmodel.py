import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np
import onnxruntime as ort
from backend.model.mobilenetv4 import mobilenetv4_conv_small
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 自定义数据集路径
data_dir = r'E:\postgraduatecode\grape_disease_monitor\img\trains_new'
test_dir = r'E:\postgraduatecode\grape_disease_monitor\img\trains'


class GrapeDiseaseModel:
    def __init__(self, data_dir, num_classes, batch_size=64, num_epochs=50, patience=10, learning_rate=0.001):
        self.data_dir = data_dir
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
        self.train_dataset, self.val_dataset = self.dataset, self.test_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # 实例化模型
        self.model = mobilenetv4_conv_small(num_classes=self.num_classes)
        self.model.to(self.device)

        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

        # 初始化早停
        self.early_stopping = EarlyStopping(patience=self.patience, delta=0.01)

    def train(self):
        print("Starting training...")
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            all_labels = []
            all_preds = []

            print(f"Starting epoch {epoch + 1}/{self.num_epochs}...")
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 梯度清零
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # 计算预测
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                accuracy = accuracy_score(all_labels, all_preds)
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(self.train_loader)}], '
                      f'Loss: {loss.item():.4f}, Running Loss: {running_loss / (i + 1):.4f}, '
                      f'Accuracy: {accuracy:.4f}')

            # 计算验证集损失
            val_loss = self.validate()
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Validation Loss: {val_loss:.4f}')

            # 更新学习率调度器
            self.scheduler.step(val_loss)

            # 检查早停条件
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                self.model.load_state_dict(self.early_stopping.best_model)
                break

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
        return val_loss

    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_preds = []
        print("Starting evaluation on test set...")
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                # 打印每个批次的测试日志
                batch_accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
                print(f'Test Batch [{i + 1}/{len(self.val_loader)}], Batch Accuracy: {batch_accuracy:.4f}')

        # 计算并打印测试集整体指标
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(
            f'Test Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}')

    def save_model_as_onnx(self, save_path):
        # 设置为评估模式
        self.model.eval()

        # 创建一个随机的输入张量作为示例
        dummy_input = torch.randn(1, 3, 128, 128, device=self.device)

        # 导出模型到ONNX
        torch.onnx.export(self.model, dummy_input, save_path, input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, opset_version=11)
        print(f'Model saved as ONNX at {save_path}')

    def load_and_infer(self, onnx_model_path, input_data):
        # 加载 ONNX 模型
        ort_session = ort.InferenceSession(onnx_model_path)

        # 准备输入数据
        inputs = {ort_session.get_inputs()[0].name: input_data}

        # 推理
        outputs = ort_session.run(None, inputs)
        return outputs[0]  # 返回第一个输出


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
    num_classes = len(os.listdir(data_dir))  # 根据数据集中的类别数来定义
    model = GrapeDiseaseModel(data_dir=data_dir, num_classes=num_classes)
    model.train()
    model.evaluate()
    #
    # # 保存为ONNX模型
    # onnx_model_path = 'grape_disease_model.onnx'
    # model.save_model_as_onnx(onnx_model_path)
    #
    # # 推理示例：假设我们有一个输入数据 (input_data) 需要推理
    # input_data = np.random.randn(1, 3, 128, 128).astype(np.float32)  # 示例输入
    # predictions = model.load_and_infer(onnx_model_path, input_data)
    # print(f"Inference result: {predictions}")
