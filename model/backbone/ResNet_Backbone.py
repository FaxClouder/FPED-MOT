import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



# --------------------- ResNet -----------------------
# BasicBlock: **kwargs可接受额外参数，bias=False在接入BN层时可节省资源
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out




class ResNet_Backbone(nn.Module):

    def __init__(self, block, layers, groups=1, width_per_group=64, norm_layer=nn.BatchNorm2d):
        super(ResNet_Backbone, self).__init__()
        self.in_channels = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.norm_layer = norm_layer

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])  # c2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # c3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # c4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # c5

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, planes * block.expansion, stride),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample,
                            groups=self.groups, width_per_group=self.width_per_group))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes,
                                groups=self.groups, width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, self.norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)  # H/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # H/4

        c2 = self.layer1(x)  # H/4
        c3 = self.layer2(c2)  # H/8
        c4 = self.layer3(c3)  # H/16
        c5 = self.layer4(c4)  # H/32

        return [c2, c3, c4, c5]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
}


# --------------------- Factory Functions -----------------------
def resnet18(pretrained=False, **kwargs):
    model = ResNet_Backbone(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet34(pretrained=False, **kwargs):
    model = ResNet_Backbone(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model

def resnet50(pretrained=False, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def resnet101(pretrained=False, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model

def resnet152(pretrained=False, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model


# --------------------- Build Backbone -----------------------
def build_backbone(model_name='resnet18', pretrained=False):
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained)
        feat_dims = [64, 128, 256, 512]
    elif model_name == 'resnet34':
        model = resnet34(pretrained=pretrained)
        feat_dims = [64, 128, 256, 512]
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        feat_dims = [256, 512, 1024, 2048]
    elif model_name == 'resnet101':
        model = resnet101(pretrained=pretrained)
        feat_dims = [256, 512, 1024, 2048]
    elif model_name == 'resnet152':
        model = resnet152(pretrained=pretrained)
        feat_dims = [256, 512, 1024, 2048]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, feat_dims