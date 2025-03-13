import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import json
from pathlib import Path


def predict_with_onnx(image_path, onnx_model_path, label_mapping_path):
    # 加载标签映射
    with open(label_mapping_path, "r", encoding="utf-8") as json_file:
        label_mapping = json.load(json_file)

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
    image = image.numpy()

    # 加载 ONNX 模型
    session = ort.InferenceSession(onnx_model_path)

    # 获取输入和输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 进行预测
    outputs = session.run([output_name], {input_name: image})[0]

    # 计算概率
    probabilities = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
    preds = np.argmax(outputs, axis=1)
    probability = probabilities[0][preds[0]]

    predicted_class = label_mapping[str(preds[0])]
    print(f"Predicted class: {predicted_class}")
    print(f"Class probability: {probability:.4f}, Image path: {image_path}")
    return predicted_class, round(probability, 4)

#
# if __name__ == "__main__":
#     image_path = r"E:\postgraduatecode\grape_disease_monitor\img\trains\溃疡病\6235845bd7561b594fb696e2.jpg"
#     onnx_model_path = "mobilenet_v4_SE_FPN.onnx"
#     label_mapping_path = str(Path(__file__).parent.joinpath("label_mapping.json").resolve())
#
#     predict_with_onnx(image_path, onnx_model_path, label_mapping_path)
