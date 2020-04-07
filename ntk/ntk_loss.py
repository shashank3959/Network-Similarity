import torch
from torch import nn

class NTKLoss(nn.Module):
    """NTK Loss function
    Computes the loss over the two feature map matrices

    Args:

    """
    def __init__(self):
        super(NTKLoss, self).__init__()

    def forward(self, kernel_s, kernel_t):
        # Compute the distance between the student and the teacher
        return torch.norm(kernel_s - kernel_t)

