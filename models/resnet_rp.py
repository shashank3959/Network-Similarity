import torch
import torch.nn as nn
from .modules import *


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
    padding=0, bias=False, grad_proj=False):
    """
    Args:
      in_channels (int): number of channels in the input.
      out_channels (int): number of channels produced by the convolution.
      kernel_size (int or tuple): size of convolving kernel.
      stride (int or tuple, optional): stride of convolution. Default: 1
      padding (int or tuple, optional): zero-padding added to both sides 
        of the input. Default: 0
      bias (bool, optional): If True, adds a learnable bias to the output.
        Default: False
      grad_proj (bool, optional): If True, projects per-sample Jacobian on
        a randomly sampled Gaussian unit vector. Default: False
     """
    super(ConvBlock, self).__init__()
    self.conv = Conv2d(in_channels, out_channels, kernel_size, stride,
      padding, bias, grad_proj)
    self.bn = BatchNorm2d(out_channels, grad_proj)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = self.conv.kernel_size
    self.stride = stride
    self.padding = padding
    self.grad_proj = grad_proj
    
    self._compute_rv_norm_sqr()

  def _compute_rv_norm_sqr(self):
    """ Returns the squared norm of the random vector. """
    if self.grad_proj:
      conv_rv_norm_sqr = self.conv.get_rv_norm_sqr()
      bn_rv_norm_sqr = self.bn.get_rv_norm_sqr()
      self.rv_norm_sqr = conv_rv_norm_sqr + bn_rv_norm_sqr
    else:
      self.register_buffer('rv_norm_sqr', None)

  def get_rv_norm_sqr(self):
    """ Returns the squared norm of the random vector. """
    return self.rv_norm_sqr

  def reset_grad_proj(self, grad_proj):
    """ Turns on/off Jacobian projection. """
    self.grad_proj = grad_proj
    self.conv.reset_grad_proj(grad_proj)
    self.bn.reset_grad_proj(grad_proj)
    self._compute_rv_norm_sqr()

  def forward(self, x, jvp=None):
    """
    Args:
      x (float, [N, C, H, W]): input.
      jvp (float, [N, C, H, W], optional): per-sample Jacobian projection from
        upstream layers. Default: None
    """
    (x_out, jvp_out) = self.bn(*self.conv(x, jvp))
    return x_out, jvp_out


class BasicBlock(nn.Module):
  expansion = 1
  def __init__(self, in_planes, planes, stride=1, downsample=None, 
    grad_proj=False):
    """
    Args:
      in_planes (int): number of channels in the input.
      planes (int): number of channels produced by the convolutions.
      stride (int or tuple, optional): stride of first convolution. Default: 1
      downsample (ConvBlock): downsampling convolution. Default: None
      grad_proj (bool, optional): If True, projects per-sample Jacobian on
        a randomly sampled Gaussian unit vector. Default: False
    """
    super(BasicBlock, self).__init__()
    self.in_planes = in_planes
    self.planes = planes
    self.stride = stride
    self.grad_proj = grad_proj

    self.conv1 = ConvBlock(
      in_planes, planes, 3, stride=stride, padding=1, grad_proj=grad_proj)
    self.conv2 = ConvBlock(
      planes, planes, 3, stride=1, padding=1, grad_proj=grad_proj)
    self.relu = ReLU(inplace=True, grad_proj=grad_proj)
    self.downsample = downsample

    self._compute_rv_norm_sqr()

  def _compute_rv_norm_sqr(self):
    """ Returns the squared norm of the random vector. """
    if self.grad_proj:
      conv1_rv_norm_sqr = self.conv1.get_rv_norm_sqr()
      conv2_rv_norm_sqr = self.conv2.get_rv_norm_sqr()
      self.rv_norm_sqr = conv1_rv_norm_sqr + conv2_rv_norm_sqr
      if self.downsample is not None:
        downsample_rv_norm_sqr = self.downsample.get_rv_norm_sqr()
        self.rv_norm_sqr += downsample_rv_norm_sqr
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
    self.relu.reset_grad_proj(grad_proj)
    if self.downsample is not None:
      self.downsample.reset_grad_proj(grad_proj)
    self._compute_rv_norm_sqr()

  def forward(self, x, jvp=None):
    """
    Args:
      x (float, [N, C, H, W]): input.
      jvp (float, [N, C, H, W], optional): per-sample Jacobian projection from
        upstream layers. Default: None
    """
    (x_out, jvp_out) = self.relu(*self.conv1(x, jvp))
    (x_out, jvp_out) = self.conv2(x_out, jvp_out)
    if self.downsample is not None:
        (x, jvp) = self.downsample(x, jvp)
    x_out = x + x_out
    if jvp is not None:
      jvp_out = jvp + jvp_out
    (x_out, jvp_out) = self.relu(x_out, jvp_out)
    return x_out, jvp_out


class Bottleneck(nn.Module):
  expansion = 4
  def __init__(self, in_planes, planes, stride=1, downsample=None,
    grad_proj=False):
    """
    Args:
      in_planes (int): number of channels in the input.
      planes (int): number of channels produced by the convolutions.
      stride (int or tuple, optional): stride of first convolution. Default: 1
      downsample (ConvBlock): downsampling convolution. Default: None
      grad_proj (bool, optional): If True, projects per-sample Jacobian on
        a randomly sampled Gaussian unit vector. Default: False
    """
    super(Bottleneck, self).__init__()
    self.in_planes = in_planes
    self.planes = planes
    self.stride = stride
    self.grad_proj = grad_proj

    self.conv1 = ConvBlock(
      in_planes, planes, 1, stride=1, grad_proj=grad_proj)
    self.conv2 = ConvBlock(
      planes, planes, 3, stride=stride, padding=1, grad_proj=grad_proj)
    self.conv3 = ConvBlock(
      planes, planes * self.expansion, 1, stride=1, grad_proj=grad_proj)
    self.relu = ReLU(inplace=True, grad_proj=grad_proj)
    self.downsample = downsample

    self._compute_rv_norm_sqr()

  def _compute_rv_norm_sqr(self):
    """ Returns the squared norm of the random vector. """
    if self.grad_proj:
      conv1_rv_norm_sqr = self.conv1.get_rv_norm_sqr()
      conv2_rv_norm_sqr = self.conv2.get_rv_norm_sqr()
      conv3_rv_norm_sqr = self.conv3.get_rv_norm_sqr()
      self.rv_norm_sqr = \
        conv1_rv_norm_sqr + conv2_rv_norm_sqr + conv3_rv_norm_sqr
      if self.downsample is not None:
        downsample_rv_norm_sqr = self.downsample.get_rv_norm_sqr()
        self.rv_norm_sqr += downsample_rv_norm_sqr
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
    self.relu.reset_grad_proj(grad_proj)
    if self.downsample is not None:
      self.downsample.reset_grad_proj(grad_proj)
    self._compute_rv_norm_sqr()

  def forward(self, x, jvp=None):
    """
    Args:
      x (float, [N, C, H, W]): input.
      jvp (float, [N, C, H, W], optional): per-sample Jacobian projection from
        upstream layers. Default: None
    """
    (x_out, jvp_out) = self.relu(*self.conv1(x, jvp))
    (x_out, jvp_out) = self.relu(*self.conv2(x_out, jvp_out))
    (x_out, jvp_out) = self.conv3(x_out, jvp_out)
    if self.downsample is not None:
        (x, jvp) = self.downsample(x, jvp)
    x_out = x + x_out
    if jvp is not None:
      jvp_out = jvp + jvp_out
    (x_out, jvp_out) = self.relu(x_out, jvp_out)
    return x_out, jvp_out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=10, grad_proj=False):
    super(ResNet, self).__init__()
    """
    Args:
      block (BasicBlock or Bottleneck): type of residual blocks.
      layers (list): number of residual blocks in each layer.
      num_classes (int): number of output dimensions. Default: 10
      grad_proj (bool, optional): If True, projects per-sample Jacobian on
        a randomly sampled Gaussian unit vector. Default: False
    """
    self.in_planes = 64
    self.grad_proj = grad_proj

    self.conv1 = ConvBlock(
      3, self.in_planes, 3, stride=1, padding=1, grad_proj=grad_proj)
    self.relu = ReLU(inplace=True, grad_proj=grad_proj)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    self.avgpool = AdaptiveAvgPool2d(1, grad_proj=grad_proj)
    self.fc = Linear(512 * block.expansion, num_classes, grad_proj=grad_proj)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    self._compute_rv_norm_sqr()

  def _make_layer(self, block, planes, blocks, stride=1):
    """ Composes residual blocks for a layer. """
    downsample = None
    if stride != 1 or self.in_planes != planes * block.expansion:
      downsample = ConvBlock(
        self.in_planes, planes * block.expansion, 1, stride=stride,
        grad_proj=self.grad_proj)
    layers = []
    layers.append(
      block(self.in_planes, planes, stride=stride, downsample=downsample,
        grad_proj=self.grad_proj))
    self.in_planes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
        block(self.in_planes, planes, grad_proj=self.grad_proj))

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
      self.rv_norm_sqr += self.fc.get_rv_norm_sqr()
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
    self.fc.reset_grad_proj(grad_proj)
    self._compute_rv_norm_sqr()

  def forward(self, x):
    """
    Args:
      x (float, [N, 3, 32, 32]): input.
    """
    (x, jvp) = self.relu(*self.conv1(x))
    (x, jvp) = self.layer1(x, jvp)
    (x, jvp) = self.layer2(x, jvp)
    (x, jvp) = self.layer3(x, jvp)
    (x, jvp) = self.layer4(x, jvp)
    (x, jvp) = self.avgpool(x, jvp)
    x = torch.flatten(x, 1)
    if jvp is not None:
      jvp = torch.flatten(jvp, 1)
    (x, jvp) = self.fc(x, jvp)
    # normalize the random vector
    if jvp is not None:
      jvp = jvp / self.rv_norm_sqr ** 0.5
    return x, jvp


def resnet18(**kwargs):
  return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
  return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
  return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)