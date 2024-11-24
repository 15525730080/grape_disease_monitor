import torch
import torch.nn as nn
import torch.nn.functional as F

from src_mobilenet_v4 import MultiHeadSelfAttentionBlock, InvertedResidual, MODEL_SPECS


# Helper functions from the original code
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
        # Squeeze
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # Excitation
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        out = self.global_avgpool(x)
        # Excitation
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        # Scale
        return x * out


# Modify the Universal Inverted Bottleneck Block to include SEBlock
class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 middle_dw_downsample,
                 stride,
                 expand_ratio,
                 use_se=False  # Added parameter to control SE
                 ):
        super().__init__()
        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        # Squeeze-and-Excitation block
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)

        # Residual connection
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


# Build blocks function to include use_se parameter
def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"convbn_{i}", conv_2d(**args))
    elif block_names == "uib":
        schema_ = ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride',
                   'expand_ratio', 'mhsa']
        for i in range(layer_spec['num_blocks']):
            args_list = layer_spec['block_specs'][i]
            args_dict = dict(zip(schema_, args_list))
            mhsa = args_dict.pop("mhsa") if "mhsa" in args_dict else None
            args_dict['use_se'] = True  # Enable SE blocks
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args_dict))
            if mhsa:
                mhsa_schema_ = [
                    "inp", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides",
                    "use_layer_scale", "use_multi_query", "use_residual"
                ]
                args_mhsa = dict(zip(mhsa_schema_, [args_dict['oup']] + mhsa))
                layers.add_module(f"mhsa_{i}", MultiHeadSelfAttentionBlock(**args_mhsa))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers


# MobileNetV4 with attention mechanisms and enhanced small object detection capabilities
class MobileNetV4Pro(nn.Module):
    def __init__(self, model, num_classes=1000):
        super().__init__()
        assert model in MODEL_SPECS.keys()
        self.model = model
        self.spec = MODEL_SPECS[self.model]

        # conv0
        self.conv0 = build_blocks(self.spec['conv0'])
        # layer1
        self.layer1 = build_blocks(self.spec['layer1'])
        # layer2
        self.layer2 = build_blocks(self.spec['layer2'])
        # layer3
        self.layer3 = build_blocks(self.spec['layer3'])
        # layer4
        self.layer4 = build_blocks(self.spec['layer4'])
        # layer5
        self.layer5 = build_blocks(self.spec['layer5'])

        # Define output channels for each layer based on the model specs
        if self.model == "MobileNetV4ConvSmall":
            self.out_channels = {
                'x1': 32,  # from layer1
                'x2': 64,  # from layer2
                'x3': 96,  # from layer3
                'x4': 128,  # from layer4
                'x5': 1280  # from layer5
            }
        elif self.model == "MobileNetV4ConvMedium":
            self.out_channels = {
                'x1': 48,
                'x2': 80,
                'x3': 160,
                'x4': 256,
                'x5': 1280
            }
        elif self.model == "MobileNetV4ConvLarge":
            self.out_channels = {
                'x1': 48,
                'x2': 96,
                'x3': 192,
                'x4': 512,
                'x5': 1280
            }
        else:
            raise NotImplementedError

        # Feature Pyramid Network (FPN) layers for small object detection
        c3_channels = self.out_channels['x2']  # x2
        c4_channels = self.out_channels['x3']  # x3
        c5_channels = self.out_channels['x4']  # x4

        # Lateral layers
        self.lateral_c5 = nn.Conv2d(c5_channels, 256, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(c4_channels, 256, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(c3_channels, 256, kernel_size=1)

        # Smooth layers
        self.smooth_p4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Classification head
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x0 = self.conv0(x)  # Output channels: depends on conv0
        x1 = self.layer1(x0)  # Output channels: x1
        x2 = self.layer2(x1)  # Output channels: x2
        x3 = self.layer3(x2)  # Output channels: x3
        x4 = self.layer4(x3)  # Output channels: x4
        x5 = self.layer5(x4)  # Output channels: x5

        # Build FPN
        # Lateral connections
        c5 = x4  # Highest level features
        c4 = x3
        c3 = x2  # Lowest level features used in FPN

        # Lateral layers
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3)

        # Top-down pathway
        p5_upsampled = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4 = p4 + p5_upsampled
        p4 = self.smooth_p4(p4)
        p4_upsampled = F.interpolate(p4, size=p3.shape[2:], mode='nearest')
        p3 = p3 + p4_upsampled
        p3 = self.smooth_p3(p3)

        # Use p3 for classification
        global_feat = F.adaptive_avg_pool2d(p3, 1).view(p3.size(0), -1)
        logits = self.classifier(global_feat)
        return logits

# Note: The MODEL_SPECS dictionary and other supporting classes/functions (e.g., InvertedResidual, MultiHeadSelfAttentionBlock)
# should be defined as in your original code or imported if they are in another module.
