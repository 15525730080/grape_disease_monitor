# 葡萄果实病害识别系统

## 项目简介

基于深度学习的葡萄果实病害识别系统。

## 模型集成能力

系统支持多种深度学习模型集成预测，以提高识别的准确性和鲁棒性。以下是系统中集成的主要模型：

    ensemble_classifier = EnsembleGrapeDiseaseClassifier(
        [
            Path(__file__).parent.joinpath("cnn_v1.keras").resolve(),
            Path(__file__).parent.joinpath("cnn_v2.keras").resolve(),
            Path(__file__).parent.joinpath("mobilenet_v2.keras").resolve()
        ],
        [
            MobilenetV4("", Path(__file__).parent.joinpath("mobilenet_v4.pth").resolve()),
            MobilenetV4SEFPN("", Path(__file__).parent.joinpath("mobilenet_v4_SE_FPN.pth").resolve()),
            MobilenetV4SEBiFPN("", Path(__file__).parent.joinpath("mobilenet_v4_SE_BiFPN.pth").resolve())
        ]
    )


## 模型优化与改进

### MobileNetV4

MobileNetV4是一个轻量级的深度学习模型，它在保持高准确率的同时，大幅减少了模型大小和计算需求。这使得模型能够在资源受限的设备上运行，如智能手机和嵌入式系统。

### MobileNetV4_SE_FPN

MobileNetV4_SE_FPN是MobileNetV4的改进版本，引入了Squeeze-and-Excitation（SE）模块和特征金字塔网络（FPN）。SE模块能够增强模型对重要特征的学习，而FPN则提高了模型对不同尺度对象的识别能力。

### MobileNetV4_SE_BiFPN

MobileNetV4_SE_BiFPN进一步扩展了MobileNetV4_SE_FPN，采用了双特征金字塔网络（BiFPN）。BiFPN通过结合来自不同层的特征，提供了更丰富的上下文信息，从而提高了模型对复杂场景下病害识别的性能。


![验证准确率对比](https://github.com/user-attachments/assets/11656316-46ff-4056-aca7-8e409039e00f)
