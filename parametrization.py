import torch
import torch.nn as nn
import torch.nn.functional as F


class NTKConv2d(nn.Module):
  """Conv2d layer under NTK parametrization."""
  def __init__(self, in_channels, out_channels, kernel_size, 
    stride=1, padding=0, bias=True):
    super(NTKConv2d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.bias = None
    self.weight = nn.Parameter(torch.randn(
      out_channels, in_channels, kernel_size, kernel_size))
    if bias:
      self.bias = nn.Parameter(torch.randn(out_channels))

  def forward(self, x, add_bias=True):
    weight = (1. / self.out_channels)**0.5 * self.weight
    if add_bias and self.bias is not None:
      bias = (.1)**0.5 * self.bias
      return F.conv2d(x, weight, bias, self.stride, self.padding)
    else:
      return F.conv2d(x, weight, None, self.stride, self.padding)


class NTKLinear(nn.Module):
  """Linear layer under NTK parametrization."""
  def __init__(self, in_features, out_features, bias=True):
    super(NTKLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features

    self.bias = None
    self.weight = nn.Parameter(torch.randn(out_features, in_features))
    if bias:
      self.bias = nn.Parameter(torch.randn(out_features))

  def forward(self, x, add_bias=True):
    weight = (1. / self.out_features)**0.5 * self.weight
    if add_bias and self.bias is not None:
      bias = (.1)**0.5 * self.bias
      return F.linear(x, weight, bias)
    else:
      return F.linear(x, weight, None)


def std_to_ntk_conv2d(conv2d):
  """STD Conv2d -> NTK Conv2d"""
  if isinstance(conv2d, NTKConv2d):
    return conv2d
  bias = True if conv2d.bias is not None else False
  ntk_conv2d = NTKConv2d(conv2d.in_channels, conv2d.out_channels, 
    conv2d.kernel_size[0], conv2d.stride, conv2d.padding, bias=bias)
  # parameter rescaling (see Jacot et al. NeurIPS 18)
  ntk_conv2d.weight.data = conv2d.weight.data / (1. / conv2d.out_channels)**0.5
  if bias:
    ntk_conv2d.bias.data = conv2d.bias.data / (.1)**0.5
  return ntk_conv2d


def ntk_to_std_conv2d(conv2d):
  """NTK Conv2d -> STD Conv2d"""
  if isinstance(conv2d, nn.Conv2d):
    return conv2d
  bias = True if conv2d.bias is not None else False
  std_conv2d = nn.Conv2d(conv2d.in_channels, conv2d.out_channels, 
    conv2d.kernel_size[0], conv2d.stride, conv2d.padding, bias=bias)
  # parameter rescaling (see Jacot et al. NeurIPS 18)
  std_conv2d.weight.data = conv2d.weight.data * (1. / conv2d.out_channels)**0.5
  if bias:
    std_conv2d.bias.data = conv2d.bias.data * (.1)**0.5
  return std_conv2d


def std_to_ntk_linear(fc):
  """STD Linear -> NTK Linear"""
  if isinstance(fc, NTKLinear):
    return fc
  bias = True if fc.bias is not None else False
  ntk_fc = NTKLinear(fc.in_features, fc.out_features)
  # parameter rescaling (see Jacot et al. NeurIPS 18)
  ntk_fc.weight.data = fc.weight.data / (1. / fc.out_features)**0.5
  if bias:
    ntk_fc.bias.data = fc.bias.data / (.1)**0.5
  return ntk_fc


def ntk_to_std_linear(fc):
  """NTK Linear -> STD Linear"""
  if isinstance(fc, NTKLinear):
    return fc
  bias = True if fc.bias is not None else False
  std_fc = NTKLinear(fc.in_features, fc.out_features)
  # parameter rescaling (see Jacot et al. NeurIPS 18)
  std_fc.weight.data = fc.weight.data * (1. / fc.out_features)**0.5
  if bias:
    std_fc.bias.data = fc.bias.data * (.1)**0.5
  return std_fc


def merge_batchnorm(conv2d, batchnorm):
  """Folds BatchNorm2d into Conv2d."""
  if isinstance(batchnorm, nn.Identity):
    return conv2d
  mean = batchnorm.running_mean
  sigma = torch.sqrt(batchnorm.running_var + batchnorm.eps)
  beta = batchnorm.weight
  gamma = batchnorm.bias

  w = conv2d.weight
  if conv2d.bias is not None:
    b = conv2d.bias
  else:
    b = torch.zeros_like(mean)

  w = w * (beta / sigma).view(conv2d.out_channels, 1, 1, 1)
  b = (b - mean) / sigma * beta + gamma
  
  fused_conv2d = nn.Conv2d(
    conv2d.in_channels, conv2d.out_channels, conv2d.kernel_size, 
    conv2d.stride, conv2d.padding)
  fused_conv2d.weight.data = w
  fused_conv2d.bias.data = b

  return fused_conv2d


def selective_pool(x, indices):
  """Pools input according to given indices."""
  flat_x = x.flatten(start_dim=2)
  flat_indices = indices.flatten(start_dim=2)
  x = flat_x.gather(dim=2, index=flat_indices).view_as(indices)
  return x