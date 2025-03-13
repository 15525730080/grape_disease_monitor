import sys
from pathlib import Path
import json
import torch
import torch.onnx
import torch.nn as nn
from pathlib import Path
sys.path.append(str(Path(__file__).parent.joinpath("model").resolve()))
# 定义 MobileNetV4 模型（确保有 mobilenet_v4.py 文件）
# 定义 MobileNetV4 模型（确保有 mobilenet_v4.py 文件）
try:
    from .model.src_mobilenet_v4_SE_FPN import MobileNetV4Pro
except:
    from model.src_mobilenet_v4_SE_FPN import MobileNetV4Pro

def convert_to_onnx(model_path, onnx_path):
    # 加载标签映射
    with open(str(Path(__file__).parent.joinpath("label_mapping.json").resolve()), "r",
              encoding="utf-8") as json_file:
        label_mapping = json.load(json_file)
    num_classes = len(label_mapping)

    # 初始化模型
    model = MobileNetV4Pro("MobileNetV4ConvSmall")
    model.fc = nn.Linear(1280, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 创建一个示例输入
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # 导出模型到 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {onnx_path}")



if __name__ == "__main__":
    model_path = str(Path(__file__).parent.joinpath("mobilenet_v4_SE_FPN.pth").resolve())
    onnx_path = "mobilenet_v4_SE_FPN.onnx"
    convert_to_onnx(model_path, onnx_path)
