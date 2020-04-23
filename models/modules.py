import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
    padding=0, bias=True, grad_proj=False):
    """ 
    Args:
      in_channels (int): number of channels in the input.
      out_channels (int): number of channels produced by the convolution.
      kernel_size (int or tuple): size of convolving kernel.
      stride (int or tuple, optional): stride of convolution. Default: 1
      padding (int or tuple, optional): zero-padding added to both sides 
        of the input. Default: 0
      bias (bool, optional): If True, adds a learnable bias to the output.
        Default: True
      grad_proj (bool, optional): If True, projects per-sample Jacobian on
        a randomly sampled Gaussian unit vector. Default: False
    """
    super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, 
      stride, padding, bias=bias)
    self.grad_proj = grad_proj
    self.reset_rv()

  def reset_rv(self):
    """ Samples a Gaussian random vector. """
    if self.grad_proj:
      self.weight_rv = torch.randn(
        self.out_channels, self.in_channels, *self.kernel_size)
      weight_rv_norm_sqr = torch.sum(self.weight_rv ** 2)
      if self.bias is not None:
        self.bias_rv = torch.randn(self.out_channels)
        bias_rv_norm_sqr = torch.sum(self.bias_rv ** 2)
      else:
        self.register_buffer('bias_rv', None)
        bias_rv_norm_sqr = 0.
      self.rv_norm_sqr = weight_rv_norm_sqr + bias_rv_norm_sqr
    else:
      self.register_buffer('weight_rv', None)
      self.register_buffer('bias_rv', None)
      self.register_buffer('rv_norm_sqr', None)

  def get_rv_norm_sqr(self):
    """ Returns squared norm of random vector. """
    return self.rv_norm_sqr

  def forward(self, x, jvp=None):
    """
    Args:
      x (float, [N, C, H, W]): input.
      jvp (float, [N, C, H, W], optional): per-sample Jacobian projection from
        upstream layers. Default: None
    """
    x_out = super(Conv2d, self).forward(x)
    if self.grad_proj:
      jvp_out = F.conv2d(
        x, self.weight_rv, self.bias_rv, self.stride, self.padding)
      if jvp is not None:
        jvp_out = jvp_out + F.conv2d(
          jvp, self.weight, None, self.stride, self.padding)
      return x_out, jvp_out
    else:
      return x_out


class Linear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, grad_proj=False):
    """
    Args:
      in_features (int): number of features in the input.
      out_features (int): number of features in the output.
      bias (bool, optional): If True, adds a learnable bias to the output.
        Default: True.
      grad_proj (bool, optional): If True, projects per-sample Jacobian on
        a randomly sampled Gaussian unit vector. Default: False
    """
    super(Linear, self).__init__(in_features, out_features, bias=bias)
    self.grad_proj = grad_proj
    self.reset_rv()

  def reset_rv(self):
    """ Samples a Gaussian random vector. """
    if self.grad_proj:
      self.weight_rv = torch.randn(self.out_features, self.in_features)
      weight_rv_norm_sqr = torch.sum(self.weight_rv ** 2)
      if self.bias is not None:
        self.bias_rv = torch.randn(self.out_features)
        bias_rv_norm_sqr = torch.sum(self.bias_rv ** 2)
      else:
        self.register_buffer('bias_rv', None)
        bias_rv_norm_sqr = 0.
      self.rv_norm_sqr = weight_rv_norm_sqr + bias_rv_norm_sqr
    else:
      self.register_buffer('weight_rv', None)
      self.register_buffer('bias_rv', None)
      self.register_buffer('rv_norm_sqr', None)

  def get_rv_norm_sqr(self):
    """ Returns squared norm of random vector. """
    return self.rv_norm_sqr

  def forward(self, x, jvp=None):
    """
    Args:
      x (float, [N, C]): input.
      jvp (float, [N, C], optional): per-sample Jacobian projection from
        upstream layers. Default: None
    """
    x_out = super(Linear, self).forward(x)
    if self.grad_proj:
      jvp_out = F.linear(x, self.weight_rv, self.bias_rv)
      if jvp is not None:
        jvp_out = jvp_out + F.linear(jvp, self.weight, None)
      return x_out, jvp_out
    else:
      return x_out


class BatchNorm2d(nn.BatchNorm2d):
  def __init__(self, num_features, eps=1e-5, momentum=0.1, grad_proj=False):
    """
    Args:
      num_features (int): number of features in the input.
      eps (float, optional): a value added to the denominator for numerical
        stability. Default: 1e-5
      momentum: a value used in the calculation of the running statistics.
        Default: 0.1
      grad_proj (bool, optional): If True, projects per-sample Jacobian on
        a randomly sampled Gaussian unit vector. Default: False
    """
    super(BatchNorm2d, self).__init__(num_features, eps, momentum)
    self.grad_proj = grad_proj
    self.reset_rv()

  def reset_rv(self):
    """ Samples a Gaussian random vector. """
    if self.grad_proj:
      self.weight_rv = torch.randn(self.num_features)
      self.bias_rv = torch.randn(self.num_features)
      weight_rv_norm_sqr = torch.sum(self.weight_rv ** 2)
      bias_rv_norm_sqr = torch.sum(self.bias_rv ** 2)
      self.rv_norm_sqr = weight_rv_norm_sqr + bias_rv_norm_sqr
    else:
      self.register_buffer('weight_rv', None)
      self.register_buffer('bias_rv', None)
      self.register_buffer('rv_norm_sqr', None)

  def get_rv_norm_sqr(self):
    """ Returns squared norm of random vector. """
    return self.rv_norm_sqr

  def batch_norm(self, x, mean, var, eps, weight, bias):
    """ Performs batch normalization. """
    inv_var = 1. / (var + eps) ** 0.5
    weight, bias = weight.view(1, -1, 1, 1), bias.view(1, -1, 1, 1)
    x_out = weight * (x - mean) * inv_var + bias
    return x_out

  def forward(self, x, jvp=None):
    """
    Args:
      x (float, [N, C, H, W]): input.
      jvp (float, [N, C, H, W], optional): per-sample Jacobian projection from
        upstream layers. Default: None
    """
    x_out = super(BatchNorm2d, self).forward(x)
    if self.grad_proj:
      if self.training:  # training mode, use mini-batch statistics
        mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        var = torch.var(x, dim=(0, 2, 3), keepdim=True)  # unbiased
        jvp_out = self.batch_norm(
          x, mean, var, self.eps, self.weight_rv, self.bias_rv)
        if jvp is not None:
          jvp = jvp - torch.mean(jvp, dim=(0, 2, 3), keepdim=True)
          jvp_out = jvp_out + self.batch_norm(
            jvp, 0., var, self.eps, self.weight, 0.)  
      else:              # evaluation mode, use running statistics
        mean, var = self.running_mean, self.running_var
        jvp_out = self.batch_norm(
          x, mean, var, self.eps, self.weight_rv, self.bias_rv)
        if jvp is not None:
          jvp_out = jvp_out + self.batch_norm(
            jvp, 0., var, self.eps, self.weight, 0.)
      return x_out, jvp_out
    else:
      return x_out


class ReLU(nn.ReLU):
  def __init__(self, inplace=False, grad_proj=False):
    """
    Args:
      inplace (bool, optional): If True, does the operation in-place.
        Default: False
      grad_proj (bool, optional): If True, projects per-sample Jacobian on
        a randomly sampled Gaussian unit vector. Default: False
    """
    super(ReLU, self).__init__(inplace)
    self.grad_proj = grad_proj

  def forward(self, x, jvp=None):
    """
    Args:
      x (float, [N, C, H, W]): input.
      jvp (float, [N, C, H, W], optional): per-sample Jacobian projection from
        upstream layers. Default: None
    """
    x_out = super(ReLU, self).forward(x)
    if self.grad_proj and jvp is not None:
      jvp_out = jvp * (x_out > 0).float()
      return x_out, jvp_out
    else:
      return x_out


class LeakyReLU(nn.LeakyReLU):
  def __init__(self, negative_slope=1e-2, inplace=False, grad_proj=False):
    """
    Args:
      negative_slope (float, optional): slope of the linear function on 
        negative input. Default: 1e-2
      inplace (bool, optional): If True, does the operation in-place.
        Default: False
      grad_proj (bool, optional): If True, projects per-sample Jacobian on
        a randomly sampled Gaussian unit vector. Default: False
    """
    super(ReLU, self).__init__(negative_slope, inplace)
    self.grad_proj = grad_proj

  def forward(self, x, jvp=None):
    """
    Args:
      x (float, [N, C, H, W]): input.
      jvp (float, [N, C, H, W], optional): per-sample Jacobian projection from
        upstream layers. Default: None
    """
    x_out = super(LeakyReLU, self).forward(x)
    if self.grad_proj and jvp is not None:
      jvp_out = jvp * (
        (x_out > 0).float() + self.negative_slope * (x_out < 0).float())
      return x_out, jvp_out
    else:
      return x_out