import torch
import torch.autograd as autograd
from modules import *


##### TEST 1: Conv2d, ReLU, AvgPool2d and Linear #####
x = torch.randn(2, 3, 8, 8)
conv1 = Conv2d(3, 8, kernel_size=3, stride=1, padding=1, grad_proj=True)
conv2 = Conv2d(8, 8, kernel_size=3, stride=2, padding=1, grad_proj=True)
relu = ReLU(inplace=True, grad_proj=True)
pool = AvgPool2d(kernel_size=4, grad_proj=True)
fc = Linear(8, 2, grad_proj=True)

# forward pass
y, jvp = pool(*relu(*conv2(*relu(*conv1(x)))))
y, jvp = fc(y.view(2, -1), jvp.view(2, -1))

# backward pass
grad = autograd.grad(y[0, 0], 
  list(conv1.parameters()) 
  + list(conv2.parameters()) 
  + list(fc.parameters()))

# parameter gradients
conv1_weight_grad, conv1_bias_grad = grad[0], grad[1]
conv2_weight_grad, conv2_bias_grad = grad[2], grad[3]
fc_weight_grad, fc_bias_grad = grad[4], grad[5]

# random vectors
conv1_weight_rv, conv1_bias_rv = conv1.weight_rv, conv1.bias_rv
conv2_weight_rv, conv2_bias_rv = conv2.weight_rv, conv2.bias_rv
fc_weight_rv, fc_bias_rv = fc.weight_rv, fc.bias_rv

# brute-force calculation of Jacobian-vector product
true_jvp = (conv1_weight_grad * conv1_weight_rv).sum() \
         + (conv1_bias_grad * conv1_bias_rv).sum() \
         + (conv2_weight_grad * conv2_weight_rv).sum() \
         + (conv2_bias_grad * conv2_bias_rv).sum() \
         + (fc_weight_grad * fc_weight_rv).sum() \
         + (fc_bias_grad * fc_bias_rv).sum()

if abs(jvp[0, 0] - true_jvp) < 1e-5:
  print('TEST 1 PASSED')
else:
  print('TEST 1 FAILED')
  print('ground truth: %.3f \t result: %.3f' % (true_jvp, jvp[0, 0]))
  exit()


##### TEST 2: Conv2d, LeakyReLU, MaxPool2d and Linear #####
x = torch.randn(2, 3, 8, 8)
conv1 = Conv2d(3, 8, kernel_size=3, stride=1, padding=1, grad_proj=True)
conv2 = Conv2d(8, 8, kernel_size=3, stride=2, padding=1, grad_proj=True)
relu = LeakyReLU(0.1, inplace=True, grad_proj=True)
pool = MaxPool2d(kernel_size=4, grad_proj=True)
fc = Linear(8, 2, grad_proj=True)

# forward pass
y, jvp = pool(*relu(*conv2(*relu(*conv1(x)))))
y, jvp = fc(y.view(2, -1), jvp.view(2, -1))

# backward pass
grad = autograd.grad(y[0, 0], 
  list(conv1.parameters()) 
  + list(conv2.parameters()) 
  + list(fc.parameters()))

# parameter gradients
conv1_weight_grad, conv1_bias_grad = grad[0], grad[1]
conv2_weight_grad, conv2_bias_grad = grad[2], grad[3]
fc_weight_grad, fc_bias_grad = grad[4], grad[5]

# random vectors
conv1_weight_rv, conv1_bias_rv = conv1.weight_rv, conv1.bias_rv
conv2_weight_rv, conv2_bias_rv = conv2.weight_rv, conv2.bias_rv
fc_weight_rv, fc_bias_rv = fc.weight_rv, fc.bias_rv

# brute-force calculation of Jacobian-vector product
true_jvp = (conv1_weight_grad * conv1_weight_rv).sum() \
         + (conv1_bias_grad * conv1_bias_rv).sum() \
         + (conv2_weight_grad * conv2_weight_rv).sum() \
         + (conv2_bias_grad * conv2_bias_rv).sum() \
         + (fc_weight_grad * fc_weight_rv).sum() \
         + (fc_bias_grad * fc_bias_rv).sum()

if abs(jvp[0, 0] - true_jvp) < 1e-5:
  print('TEST 2 PASSED')
else:
  print('TEST 2 FAILED')
  print('ground truth: %.3f \t result: %.3f' % (true_jvp, jvp[0, 0]))
  exit()


##### TEST 3: Conv2d, BatchNorm2d (train), ReLU, AvgPool2d and Linear #####
x = torch.randn(2, 3, 8, 8)
conv1 = Conv2d(3, 8, kernel_size=3, stride=1, padding=1, grad_proj=True)
bn1 = BatchNorm2d(8, eps=0, grad_proj=True)
conv2 = Conv2d(8, 8, kernel_size=3, stride=2, padding=1, grad_proj=True)
bn2 = BatchNorm2d(8, eps=0, grad_proj=True)
relu = ReLU(inplace=True, grad_proj=True)
pool = AvgPool2d(kernel_size=4, grad_proj=True)
fc = Linear(8, 2, grad_proj=True)

# set to training mode
bn1.train()
bn2.train()

# forward pass
y, jvp = pool(*relu(*bn2(*conv2(*relu(*bn1(*conv1(x)))))))
y, jvp = fc(y.view(2, -1), jvp.view(2, -1))

# backward pass
grad = autograd.grad(y[0, 0], 
  list(conv1.parameters()) + list(bn1.parameters())
  + list(conv2.parameters()) + list(bn2.parameters())
  + list(fc.parameters()))

# parameter gradients
conv1_weight_grad, conv1_bias_grad = grad[0], grad[1]
bn1_weight_grad, bn1_bias_grad = grad[2], grad[3]
conv2_weight_grad, conv2_bias_grad = grad[4], grad[5]
bn2_weight_grad, bn2_bias_grad = grad[6], grad[7]
fc_weight_grad, fc_bias_grad = grad[8], grad[9]

# random vectors
conv1_weight_rv, conv1_bias_rv = conv1.weight_rv, conv1.bias_rv
bn1_weight_rv, bn1_bias_rv = bn1.weight_rv, bn1.bias_rv
conv2_weight_rv, conv2_bias_rv = conv2.weight_rv, conv2.bias_rv
bn2_weight_rv, bn2_bias_rv = bn2.weight_rv, bn2.bias_rv
fc_weight_rv, fc_bias_rv = fc.weight_rv, fc.bias_rv

# brute-force calculation of Jacobian-vector product
true_jvp = (conv1_weight_grad * conv1_weight_rv).sum() \
         + (conv1_bias_grad * conv1_bias_rv).sum() \
         + (bn1_weight_grad * bn1_weight_rv).sum() \
         + (bn1_bias_grad * bn1_bias_rv).sum() \
         + (conv2_weight_grad * conv2_weight_rv).sum() \
         + (conv2_bias_grad * conv2_bias_rv).sum() \
         + (bn2_weight_grad * bn2_weight_rv).sum() \
         + (bn2_bias_grad * bn2_bias_rv).sum() \
         + (fc_weight_grad * fc_weight_rv).sum() \
         + (fc_bias_grad * fc_bias_rv).sum()

if abs(jvp[0, 0] - true_jvp) < 1e-5:
  print('TEST 3 PASSED')
else:
  print('TEST 3 FAILED')
  print('ground truth: %.3f \t result: %.3f' % (true_jvp, jvp[0, 0]))
  # exit()


##### TEST 4: Conv2d, BatchNorm2d (eval), LeakyReLU, MaxPool2d and Linear #####
x = torch.randn(2, 3, 8, 8)
conv1 = Conv2d(3, 8, kernel_size=3, stride=1, padding=1, grad_proj=True)
bn1 = BatchNorm2d(8, grad_proj=True)
conv2 = Conv2d(8, 8, kernel_size=3, stride=2, padding=1, grad_proj=True)
bn2 = BatchNorm2d(8, grad_proj=True)
relu = LeakyReLU(0.1, inplace=True, grad_proj=True)
pool = MaxPool2d(kernel_size=4, grad_proj=True)
fc = Linear(8, 2, grad_proj=True)

# set to evaluation mode
bn1.eval()
bn2.eval()

# forward pass
y, jvp = pool(*relu(*bn2(*conv2(*relu(*bn1(*conv1(x)))))))
y, jvp = fc(y.view(2, -1), jvp.view(2, -1))

# backward pass
grad = autograd.grad(y[0, 0], 
  list(conv1.parameters()) + list(bn1.parameters())
  + list(conv2.parameters()) + list(bn2.parameters())
  + list(fc.parameters()))

# parameter gradients
conv1_weight_grad, conv1_bias_grad = grad[0], grad[1]
bn1_weight_grad, bn1_bias_grad = grad[2], grad[3]
conv2_weight_grad, conv2_bias_grad = grad[4], grad[5]
bn2_weight_grad, bn2_bias_grad = grad[6], grad[7]
fc_weight_grad, fc_bias_grad = grad[8], grad[9]

# random vectors
conv1_weight_rv, conv1_bias_rv = conv1.weight_rv, conv1.bias_rv
bn1_weight_rv, bn1_bias_rv = bn1.weight_rv, bn1.bias_rv
conv2_weight_rv, conv2_bias_rv = conv2.weight_rv, conv2.bias_rv
bn2_weight_rv, bn2_bias_rv = bn2.weight_rv, bn2.bias_rv
fc_weight_rv, fc_bias_rv = fc.weight_rv, fc.bias_rv

# brute-force calculation of Jacobian-vector product
true_jvp = (conv1_weight_grad * conv1_weight_rv).sum() \
         + (conv1_bias_grad * conv1_bias_rv).sum() \
         + (bn1_weight_grad * bn1_weight_rv).sum() \
         + (bn1_bias_grad * bn1_bias_rv).sum() \
         + (conv2_weight_grad * conv2_weight_rv).sum() \
         + (conv2_bias_grad * conv2_bias_rv).sum() \
         + (bn2_weight_grad * bn2_weight_rv).sum() \
         + (bn2_bias_grad * bn2_bias_rv).sum() \
         + (fc_weight_grad * fc_weight_rv).sum() \
         + (fc_bias_grad * fc_bias_rv).sum()

if abs(jvp[0, 0] - true_jvp) < 1e-5:
  print('TEST 4 PASSED')
else:
  print('TEST 4 FAILED')
  print('ground truth: %.3f \t result: %.3f' % (true_jvp, jvp[0, 0]))
  exit()