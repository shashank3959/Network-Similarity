'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
# import torch.nn.functional as F
from .modules import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False, grad_proj=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.grad_proj = grad_proj
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, grad_proj=grad_proj)
        self.bn1 = BatchNorm2d(planes, grad_proj)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, grad_proj=grad_proj)
        self.bn2 = BatchNorm2d(planes, grad_proj)
        self.relu = ReLU(inplace=True, grad_proj=grad_proj)


        # self.shortcut = nn.Sequential()
        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut, jvp = Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, grad_proj=grad_proj),
                BatchNorm2d(self.expansion * planes, grad_proj)
            )

        self._compute_rv_norm_sqr()

    def _compute_rv_norm_sqr(self):
        """ Returns the squared norm of the random vector. """
        if self.grad_proj:
            conv1_rv_norm_sqr = self.conv1.get_rv_norm_sqr()
            conv2_rv_norm_sqr = self.conv2.get_rv_norm_sqr()
            bn1_rv_norm_sqr = self.bn1.get_rv_norm_sqr()
            bn2_rv_norm_sqr = self.bn2.get_rv_norm_sqr()
            self.rv_norm_sqr = conv1_rv_norm_sqr + conv2_rv_norm_sqr + bn1_rv_norm_sqr + \
                               bn2_rv_norm_sqr
        else:
            self.register_buffer('rv_norm_sqr', None)

    def get_rv_norm_sqr(self):
        """ Returns the squared norm of the random vector. """
        return self.rv_norm_sqr

    def reset_grad_proj(self, grad_proj):
        """ Turns on/off Jacobian projection. """
        self.grad_proj = grad_proj
        self.conv1.reset_grad_proj(grad_proj)
        self.conv2.reset_grad_proj(grad_proj)
        self.bn1.reset_grad_proj(grad_proj)
        self.bn2.reset_grad_proj(grad_proj)
        self.relu.reset_grad_proj(grad_proj)
        self._compute_rv_norm_sqr()

    def forward(self, x, jvp=None):
        """
        Args:
          x (float, [N, C, H, W]): input.
          jvp (float, [N, C, H, W], optional): per-sample Jacobian projection from
            upstream layers. Default: None
        """
        (x_out, jvp_out) = self.relu(*self.bn1(*self.conv1(x, jvp)))
        (x_out, jvp_out) = self.bn2(*self.conv2(x_out, jvp_out))
        x_short, jvp_short = self.shortcut(x, jvp)
        x_out += x_short
        if self.grad_proj:
            jvp_out += jvp_short
        preact = x_out
        (x_out, jvp_out) = self.relu(x_out, jvp_out)
        if self.is_last: # Change required?
            return x_out, preact
        else:
            return x_out, jvp_out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False, grad_proj=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.grad_proj = grad_proj
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False, grad_proj=grad_proj)
        self.bn1 = BatchNorm2d(planes, grad_proj=grad_proj)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                            grad_proj=grad_proj)
        self.bn2 = BatchNorm2d(planes, grad_proj=grad_proj)
        self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False, grad_proj=grad_proj)
        self.bn3 = BatchNorm2d(self.expansion * planes, grad_proj=grad_proj)
        self.relu = ReLU(inplace=True, grad_proj=grad_proj)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                       grad_proj=grad_proj),
                BatchNorm2d(self.expansion * planes, grad_proj=grad_proj)
            )

        self._compute_rv_norm_sqr()

    def _compute_rv_norm_sqr(self):
        """ Returns the squared norm of the random vector. """
        if self.grad_proj:
            conv1_rv_norm_sqr = self.conv1.get_rv_norm_sqr()
            conv2_rv_norm_sqr = self.conv2.get_rv_norm_sqr()
            conv3_rv_norm_sqr = self.conv3.get_rv_norm_sqr()
            bn1_rv_norm_sqr = self.bn1.get_rv_norm_sqr()
            bn2_rv_norm_sqr = self.bn2.get_rv_norm_sqr()
            bn3_rv_norm_sqr = self.bn3.get_rv_norm_sqr()
            self.rv_norm_sqr = \
            conv1_rv_norm_sqr + conv2_rv_norm_sqr + conv3_rv_norm_sqr + \
            bn1_rv_norm_sqr + bn2_rv_norm_sqr + bn3_rv_norm_sqr
        else:
            self.register_buffer('rv_norm_sqr', None)

    def get_rv_norm_sqr(self):
        """ Returns the squared norm of the random vector. """
        return self.rv_norm_sqr

    def reset_grad_proj(self, grad_proj):
        """ Turns on/off Jacobian projection. """
        self.grad_proj = grad_proj
        self.conv1.reset_grad_proj(grad_proj)
        self.conv2.reset_grad_proj(grad_proj)
        self.conv3.reset_grad_proj(grad_proj)
        self.bn1.reset_grad_proj(grad_proj)
        self.bn2.reset_grad_proj(grad_proj)
        self.bn3.reset_grad_proj(grad_proj)
        self.relu.reset_grad_proj(grad_proj)
        self._compute_rv_norm_sqr()


    def forward(self, x, jvp=None):
        (x_out, jvp_out) = self.relu(*self.bn1(*self.conv1(x, jvp)))
        (x_out, jvp_out) = self.relu(*self.bn2(*self.conv2(x_out, jvp_out)))
        (x_out, jvp_out) = self.bn3(*self.conv3(x_out, jvp_out))

        x_short, jvp_short = self.shortcut(x)
        x_out += x_short
        if self.grad_proj:
            jvp_out += jvp_short

        preact = x_out

        x_out, jvp_out = self.relu(x_out, jvp_out)

        if self.is_last: # Change required?
            return x_out, preact
        else:
            return x_out, jvp_out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False,
                 grad_proj=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.grad_proj = grad_proj

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, grad_proj=self.grad_proj)
        self.bn1 = BatchNorm2d(64, self.grad_proj)
        self.relu = ReLU(inplace=True, grad_proj=grad_proj)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1), grad_proj=grad_proj)
        self.linear = Linear(512 * block.expansion, num_classes, grad_proj=grad_proj)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self._compute_rv_norm_sqr()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m

    # WARNING: Does this return a tuple now?
    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def _compute_rv_norm_sqr(self):
        """ Returns the squared norm of the random vector. """
        if self.grad_proj:
            self.rv_norm_sqr = self.conv1.get_rv_norm_sqr()
            for l in self.layer1:
                self.rv_norm_sqr += l.get_rv_norm_sqr()
            for l in self.layer2:
                self.rv_norm_sqr += l.get_rv_norm_sqr()
            for l in self.layer3:
                self.rv_norm_sqr += l.get_rv_norm_sqr()
            for l in self.layer4:
                self.rv_norm_sqr += l.get_rv_norm_sqr()
            self.rv_norm_sqr += self.linear.get_rv_norm_sqr()
        else:
            self.register_buffer('rv_norm_sqr', None)

    def get_rv_norm_sqr(self):
        """ Returns the squared norm of the random vector. """
        return self.rv_norm_sqr

    def reset_grad_proj(self, grad_proj):
        """ Turns on/off Jacobian projection. """
        self.grad_proj = grad_proj
        self.conv1.reset_grad_proj(grad_proj)
        for l in self.layer1:
            l.reset_grad_proj(grad_proj)
        for l in self.layer2:
            l.reset_grad_proj(grad_proj)
        for l in self.layer3:
            l.reset_grad_proj(grad_proj)
        for l in self.layer4:
            l.reset_grad_proj(grad_proj)
        self.relu.reset_grad_proj(grad_proj)
        self.avgpool.reset_grad_proj(grad_proj)
        self.linear.reset_grad_proj(grad_proj)
        self._compute_rv_norm_sqr()

    # WARNING: Preactivation based method will not work now
    def forward(self, x, is_feat=False, preact=False):
        (x_out, jvp) = self.relu(*self.bn1(*self.conv1(x)))
        f0 = x_out
        (x_out, jvp) = self.layer1(x_out, jvp)
        f1 = x_out
        (x_out, jvp) = self.layer2(x_out, jvp)
        f2 = x_out
        (x_out, jvp) = self.layer3(x_out, jvp)
        f3 = x_out
        (x_out, jvp) = self.layer4(x_out, jvp)
        f4 = x_out
        (x_out, jvp) = self.avgpool(x_out, jvp)
        x_out = x_out.view(x_out.size(0), -1)
        f5 = x_out
        (x_out, jvp) = self.linear(x_out, jvp)
        if is_feat:
            if preact:
                return [[f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], out]
            else:
                return [f0, f1, f2, f3, f4, f5], x_out
        else:
            return (x_out, jvp)


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    net = ResNet18(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
