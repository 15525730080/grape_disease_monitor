import re
import matplotlib
import matplotlib.pyplot as plt

# 设置支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或者使用其他支持中文的字体名称
matplotlib.rcParams['axes.unicode_minus'] = False

# 第一步：读取日志文件
with open("mobilenet_v4.log", "r", encoding='utf-8') as f:
    base_log = f.read()
with open("mobilenet_v4_SE_FPN.log", "r", encoding='utf-8') as f:
    FPN_log = f.read()
with open("mobilenet_v4_SE_BiFPN.log", "r", encoding='utf-8') as f:
    provide_log = f.read()

# 第二步：解析日志内容
# 定义正则表达式模板来提取所需数据
pattern = r"Epoch (\d+)/\d+\n.*?- Learning rate: [\d\.e-]+\n.*?- Training loss: ([\d\.e-]+), accuracy: ([\d\.e-]+)\n.*?- Validation loss: ([\d\.e-]+), accuracy: ([\d\.e-]+)"


def parse_log(log_content):
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    matches = re.findall(pattern, log_content)
    for match in matches:
        epoch = int(match[0])
        train_loss = float(match[1])
        train_acc = float(match[2])
        val_loss = float(match[3])
        val_acc = float(match[4])

        epochs.append(epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    return epochs, train_losses, train_accuracies, val_losses, val_accuracies


# 解析日志
base_epochs, base_train_losses, base_train_accuracies, base_val_losses, base_val_accuracies = parse_log(base_log)
provide_epochs, provide_train_losses, provide_train_accuracies, provide_val_losses, provide_val_accuracies = parse_log(
    provide_log)
FPN_epochs, FPN_train_losses, FPN_train_accuracies, FPN_val_losses, FPN_val_accuracies = parse_log(FPN_log)

# 第三步：绘制曲线图

# 1. 绘制损失曲线
plt.figure(figsize=(12, 8))

# 绘制基础模型
plt.plot(base_epochs, base_train_losses, label='MobileNetV4-训练损失', marker='o', linestyle='-')
plt.plot(base_epochs, base_val_losses, label='MobileNetV4-验证损失', marker='o', linestyle='--')

# 绘制增强SE-FPN模型
plt.plot(provide_epochs, provide_train_losses, label='MobileNetV4-SE-BiFPN-训练损失', marker='s', linestyle='-')
plt.plot(provide_epochs, provide_val_losses, label='MobileNetV4-SE-BiFPN-验证损失', marker='s', linestyle='--')

# 绘制增强SE-BiFPN模型
plt.plot(FPN_epochs, FPN_train_losses, label='MobileNetV4-SE-FPN-训练损失', marker='^', linestyle='-')
plt.plot(FPN_epochs, FPN_val_losses, label='MobileNetV4-SE-FPN-验证损失', marker='^', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.title('训练和验证损失曲线对比')

# 设置Y轴范围，可以根据实际数据调整
all_losses = (base_train_losses + base_val_losses +
              provide_train_losses + provide_val_losses +
              FPN_train_losses + FPN_val_losses)

plt.ylim([
    min(all_losses) * 0.9,
    max(all_losses) * 1.1
])

plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()  # 开启次要刻度
plt.savefig('loss_comparison.png', dpi=300)
plt.show()

# 2. 绘制准确率曲线
plt.figure(figsize=(12, 8))

# 绘制基础模型
plt.plot(base_epochs, base_train_accuracies, label='MobileNetV4-训练准确率', marker='o', linestyle='-')
plt.plot(base_epochs, base_val_accuracies, label='MobileNetV4-验证准确率', marker='o', linestyle='--')

# 绘制增强SE-FPN模型
plt.plot(provide_epochs, provide_train_accuracies, label='MobileNetV4-SE-BiFPN-训练准确率', marker='s', linestyle='-')
plt.plot(provide_epochs, provide_val_accuracies, label='MobileNetV4-SE-BiFPN-验证准确率', marker='s', linestyle='--')

# 绘制增强SE-BiFPN模型
plt.plot(FPN_epochs, FPN_train_accuracies, label='MobileNetV4-SE-FPN-训练准确率', marker='^', linestyle='-')
plt.plot(FPN_epochs, FPN_val_accuracies, label='MobileNetV4-SE-FPN-验证准确率', marker='^', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.title('训练和验证准确率曲线对比')

# 设置Y轴范围，放大差异
all_accuracies = (base_train_accuracies + base_val_accuracies +
                  provide_train_accuracies + provide_val_accuracies +
                  FPN_train_accuracies + FPN_val_accuracies)

# 根据实际数据设置Y轴范围，比如准确率在90%-100%之间
plt.ylim([
    min(all_accuracies) * 0.98,  # 比最小值小2%
    max(all_accuracies) * 1.002  # 比最大值大0.2%
])

plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.savefig('accuracy_comparison.png', dpi=300)
plt.show()

# 3. 绘制验证准确率的差异曲线（可选）
plt.figure(figsize=(12, 8))

# 基础模型与增强模型的验证准确率差异
# 为了计算差异，需要确保列表长度一致
min_length = min(len(base_val_accuracies), len(provide_val_accuracies), len(FPN_val_accuracies))

# 截取相同长度的数据
base_val_accuracies = base_val_accuracies[:min_length]
provide_val_accuracies = provide_val_accuracies[:min_length]
FPN_val_accuracies = FPN_val_accuracies[:min_length]
epochs = base_epochs[:min_length]

diff_provide = [p - b for p, b in zip(provide_val_accuracies, base_val_accuracies)]
diff_FPN = [f - b for f, b in zip(FPN_val_accuracies, base_val_accuracies)]

plt.plot(epochs, diff_provide, label='SE-BiFPN vs Base 验证准确率差异', marker='s', linestyle='-')
plt.plot(epochs, diff_FPN, label='SE-FPN vs Base 验证准确率差异', marker='^', linestyle='-')

plt.axhline(0, color='gray', linewidth=0.5)  # 添加水平线 y=0
plt.xlabel('Epoch')
plt.ylabel('准确率差异')
plt.title('增强模型与基础模型验证准确率差异')

plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.savefig('accuracy_difference.png', dpi=300)
plt.show()
