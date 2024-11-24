import torch
import torch.nn as nn
import torch.nn.functional as F

from .src_mobilenet_v4 import MultiHeadSelfAttentionBlock, InvertedResidual, MODEL_SPECS, build_blocks


# Helper functions
def make_divisible(value: float, divisor: int, min_value: float = None, round_down_protect: bool = True) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6(inplace=True))
    return conv


# Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out


# Universal Inverted Bottleneck Block
class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, inp, oup, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample, stride,
                 expand_ratio, use_se=False):
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(expand_filters)
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)
        self.use_res_connect = (stride == 1 and inp == oup)

    def forward(self, x):
        identity = x
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
        x = self._expand_conv(x)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
        if self.use_se:
            x = self.se(x)
        x = self._proj_conv(x)
        if self.use_res_connect:
            x = x + identity
        return x


# Bi-Directional Feature Pyramid Network (BiFPN)
class BiFPN(nn.Module):
    def __init__(self, input_channels, feature_size=256, num_layers=3):
        super(BiFPN, self).__init__()
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.feature_size = feature_size

        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, feature_size, kernel_size=1) for in_channels in input_channels
        ])
        self.upsample_convs = nn.ModuleList([
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1) for _ in range(num_layers - 1)
        ])
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1) for _ in range(num_layers - 1)
        ])
        self.weights = nn.Parameter(torch.ones((2, num_layers - 1)), requires_grad=True)

    def forward(self, features):
        lateral_feats = [conv(f) for f, conv in zip(features, self.lateral_convs)]
        # Top-down pathway
        for i in range(len(lateral_feats) - 1, 0, -1):
            upsampled = F.interpolate(lateral_feats[i], size=lateral_feats[i - 1].shape[2:], mode='nearest')
            w1, w2 = F.relu(self.weights[:, i - 1], inplace=False)
            lateral_feats[i - 1] = (w1 * lateral_feats[i - 1] + w2 * upsampled) / (w1 + w2 + 1e-6)
            lateral_feats[i - 1] = self.upsample_convs[i - 1](lateral_feats[i - 1])
        # Bottom-up pathway
        for i in range(len(lateral_feats) - 1):
            downsampled = F.interpolate(lateral_feats[i], size=lateral_feats[i + 1].shape[2:], mode='nearest')
            w1, w2 = F.relu(self.weights[:, i], inplace=False)
            lateral_feats[i + 1] = (w1 * lateral_feats[i + 1] + w2 * downsampled) / (w1 + w2 + 1e-6)
            lateral_feats[i + 1] = self.downsample_convs[i](lateral_feats[i + 1])
        return lateral_feats


# MobileNetV4Pro with BiFPN
class MobileNetV4Pro(nn.Module):
    def __init__(self, model, num_classes=1000):
        super().__init__()
        assert model in MODEL_SPECS.keys()
        self.model = model
        self.spec = MODEL_SPECS[self.model]

        self.conv0 = build_blocks(self.spec['conv0'])
        self.layer1 = build_blocks(self.spec['layer1'])
        self.layer2 = build_blocks(self.spec['layer2'])
        self.layer3 = build_blocks(self.spec['layer3'])
        self.layer4 = build_blocks(self.spec['layer4'])
        self.layer5 = build_blocks(self.spec['layer5'])

        # Define output channels for each layer based on the model specs
        self.out_channels = {
            'x1': 32,
            'x2': 64,
            'x3': 96,
            'x4': 128,
            'x5': 1280
        }
        self.bifpn = BiFPN([self.out_channels['x2'], self.out_channels['x3'], self.out_channels['x4']],
                           feature_size=256)

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.bifpn([x2, x3, x4])
        global_feat = F.adaptive_avg_pool2d(features[0], 1).view(features[0].size(0), -1)
        logits = self.classifier(global_feat)
        return logits
