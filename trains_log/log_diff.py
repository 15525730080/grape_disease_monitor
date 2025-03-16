import re
import matplotlib
import matplotlib.pyplot as plt
# 设置支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# 读取日志文件
def read_log(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return f.read()


# 解析日志内容
def parse_log(log_content):
    pattern = r"Epoch (\d+)/\d+\n.*?- Learning rate: ([\d\.e-]+)\n.*?- Training loss: ([\d\.e-]+), accuracy: ([\d\.e-]+)\n.*?- Validation loss: ([\d\.e-]+), accuracy: ([\d\.e-]+)"
    epochs = []
    learning_rates = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    matches = re.findall(pattern, log_content)
    for match in matches:
        epoch = int(match[0])
        lr = float(match[1])
        train_loss = float(match[2])
        train_acc = float(match[3])
        val_loss = float(match[4])
        val_acc = float(match[5])
        epochs.append(epoch)
        learning_rates.append(lr)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    return epochs, learning_rates, train_losses, train_accuracies, val_losses, val_accuracies


# 读取并解析日志
mobilenet_log = read_log("mobilenet_v4.log")
mobilenet_epochs, mobilenet_lrs, mobilenet_train_losses, mobilenet_train_accuracies, mobilenet_val_losses, mobilenet_val_accuracies = parse_log(
    mobilenet_log)

mobilenet_se_fpn_log = read_log("mobilenet_v4_SE_FPN.log")
mobilenet_se_fpn_epochs, mobilenet_se_fpn_lrs, mobilenet_se_fpn_train_losses, mobilenet_se_fpn_train_accuracies, mobilenet_se_fpn_val_losses, mobilenet_se_fpn_val_accuracies = parse_log(
    mobilenet_se_fpn_log)

mobilenet_se_bifpn_log = read_log("mobilenet_v4_SE_BiFPN.log")
mobilenet_se_bifpn_epochs, mobilenet_se_bifpn_lrs, mobilenet_se_bifpn_train_losses, mobilenet_se_bifpn_train_accuracies, mobilenet_se_bifpn_val_losses, mobilenet_se_bifpn_val_accuracies = parse_log(
    mobilenet_se_bifpn_log)

ghostnet_log = read_log("GhostNet_v2.log")
ghostnet_epochs, ghostnet_lrs, ghostnet_train_losses, ghostnet_train_accuracies, ghostnet_val_losses, ghostnet_val_accuracies = parse_log(
    ghostnet_log)

efficientnet_log = read_log("EfficientNet-B7.log")
efficientnet_epochs, efficientnet_lrs, efficientnet_train_losses, efficientnet_train_accuracies, efficientnet_val_losses, efficientnet_val_accuracies = parse_log(
    efficientnet_log)


# 绘制对比曲线和差异曲线的函数
def plot_comparison_and_difference(epochs1, val_losses1, val_accuracies1, lrs1,
                                   epochs2, val_losses2, val_accuracies2, lrs2,
                                   epochs3, val_losses3, val_accuracies3, lrs3,
                                   label1, label2, label3, title_prefix):
    # 绘制验证损失对比曲线
    plt.figure(figsize=(12, 8))
    plt.plot(epochs1, val_losses1, label=label1 + '-验证损失', marker='o', linestyle='-')
    plt.plot(epochs2, val_losses2, label=label2 + '-验证损失', marker='s', linestyle='-')
    plt.plot(epochs3, val_losses3, label=label3 + '-验证损失', marker='^', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('验证损失值')
    plt.title(title_prefix + '验证损失曲线对比')
    all_losses = val_losses1 + val_losses2 + val_losses3
    plt.ylim([min(all_losses) * 0.9, max(all_losses) * 1.1])
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig(title_prefix + 'val_loss_comparison.png', dpi=300)
    plt.show()

    # 绘制验证准确率对比曲线
    plt.figure(figsize=(12, 8))
    plt.plot(epochs1, val_accuracies1, label=label1 + '-验证准确率', marker='o', linestyle='-')
    plt.plot(epochs2, val_accuracies2, label=label2 + '-验证准确率', marker='s', linestyle='-')
    plt.plot(epochs3, val_accuracies3, label=label3 + '-验证准确率', marker='^', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('验证准确率')
    plt.title(title_prefix + '验证准确率曲线对比')
    all_accuracies = val_accuracies1 + val_accuracies2 + val_accuracies3
    plt.ylim([min(all_accuracies) * 0.98, max(all_accuracies) * 1.002])
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig(title_prefix + 'val_accuracy_comparison.png', dpi=300)
    plt.show()

    # 绘制验证学习率对比曲线
    plt.figure(figsize=(12, 8))
    plt.plot(epochs1, lrs1, label=label1 + '-验证学习率', marker='o', linestyle='-')
    plt.plot(epochs2, lrs2, label=label2 + '-验证学习率', marker='s', linestyle='-')
    plt.plot(epochs3, lrs3, label=label3 + '-验证学习率', marker='^', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('验证学习率')
    plt.title(title_prefix + '验证学习率曲线对比')
    all_lrs = lrs1 + lrs2 + lrs3
    plt.ylim([min(all_lrs) * 0.9, max(all_lrs) * 1.1])
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig(title_prefix + 'val_learning_rate_comparison.png', dpi=300)
    plt.show()

    # 绘制验证损失差异曲线
    plt.figure(figsize=(12, 8))
    min_length_loss = min(len(val_losses1), len(val_losses2), len(val_losses3))
    epochs_loss = epochs1[:min_length_loss]
    val_losses1_short = val_losses1[:min_length_loss]
    val_losses2_short = val_losses2[:min_length_loss]
    val_losses3_short = val_losses3[:min_length_loss]

    diff_2_1_loss = [v2 - v1 for v2, v1 in zip(val_losses2_short, val_losses1_short)]
    diff_3_1_loss = [v3 - v1 for v3, v1 in zip(val_losses3_short, val_losses1_short)]

    plt.plot(epochs_loss, diff_2_1_loss, label=label2 + ' vs ' + label1 + '验证损失差异', marker='s', linestyle='-')
    plt.plot(epochs_loss, diff_3_1_loss, label=label3 + ' vs ' + label1 + '验证损失差异', marker='^', linestyle='-')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('验证损失差异')
    plt.title(title_prefix + '验证损失差异')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig(title_prefix + 'val_loss_difference.png', dpi=300)
    plt.show()

    # 绘制验证准确率差异曲线
    plt.figure(figsize=(12, 8))
    min_length_acc = min(len(val_accuracies1), len(val_accuracies2), len(val_accuracies3))
    epochs_acc = epochs1[:min_length_acc]
    val_accuracies1_short = val_accuracies1[:min_length_acc]
    val_accuracies2_short = val_accuracies2[:min_length_acc]
    val_accuracies3_short = val_accuracies3[:min_length_acc]

    diff_2_1_acc = [v2 - v1 for v2, v1 in zip(val_accuracies2_short, val_accuracies1_short)]
    diff_3_1_acc = [v3 - v1 for v3, v1 in zip(val_accuracies3_short, val_accuracies1_short)]

    plt.plot(epochs_acc, diff_2_1_acc, label=label2 + ' vs ' + label1 + '验证准确率差异', marker='s', linestyle='-')
    plt.plot(epochs_acc, diff_3_1_acc, label=label3 + ' vs ' + label1 + '验证准确率差异', marker='^', linestyle='-')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('验证准确率差异')
    plt.title(title_prefix + '验证准确率差异')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig(title_prefix + 'val_accuracy_difference.png', dpi=300)
    plt.show()

    # 绘制验证学习率差异曲线
    plt.figure(figsize=(12, 8))
    min_length_lr = min(len(lrs1), len(lrs2), len(lrs3))
    epochs_lr = epochs1[:min_length_lr]
    lrs1_short = lrs1[:min_length_lr]
    lrs2_short = lrs2[:min_length_lr]
    lrs3_short = lrs3[:min_length_lr]

    diff_2_1_lr = [v2 - v1 for v2, v1 in zip(lrs2_short, lrs1_short)]
    diff_3_1_lr = [v3 - v1 for v3, v1 in zip(lrs3_short, lrs1_short)]

    plt.plot(epochs_lr, diff_2_1_lr, label=label2 + ' vs ' + label1 + '验证学习率差异', marker='s', linestyle='-')
    plt.plot(epochs_lr, diff_3_1_lr, label=label3 + ' vs ' + label1 + '验证学习率差异', marker='^', linestyle='-')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('验证学习率差异')
    plt.title(title_prefix + '验证学习率差异')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig(title_prefix + 'val_learning_rate_difference.png', dpi=300)
    plt.show()


# MobileNetV4优化对比
plot_comparison_and_difference(mobilenet_epochs, mobilenet_val_losses, mobilenet_val_accuracies, mobilenet_lrs,
                               mobilenet_se_fpn_epochs, mobilenet_se_fpn_val_losses, mobilenet_se_fpn_val_accuracies,
                               mobilenet_se_fpn_lrs,
                               mobilenet_se_bifpn_epochs, mobilenet_se_bifpn_val_losses, mobilenet_se_bifpn_val_accuracies,
                               mobilenet_se_bifpn_lrs,
                               'MobileNetV4', 'MobileNetV4-SE-FPN', 'MobileNetV4-SE-BiFPN', 'MobileNetV4优化：')

# 基座模型对比
plot_comparison_and_difference(mobilenet_epochs, mobilenet_val_losses, mobilenet_val_accuracies, mobilenet_lrs,
                               ghostnet_epochs, ghostnet_val_losses, ghostnet_val_accuracies, ghostnet_lrs,
                               efficientnet_epochs, efficientnet_val_losses, efficientnet_val_accuracies, efficientnet_lrs,
                               'MobileNetV4', 'GhostNet_v2', 'EfficientNet-B7', '基座模型：')