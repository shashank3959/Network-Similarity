from __future__ import print_function

import torch
import numpy as np
from functools import reduce

def compute_rowwise_dot(mat, new_mat):
    """
    Computes row-row dot products of matrix. Resultant matrix of size (batch-size x batch-size)
    """
    batch_size = mat.shape[0]
    new_mat = new_mat.new_zeros((batch_size, batch_size))

    # Calculate only the upper triangular part to save on half calculations
    for i in range(batch_size):
        for j in range(i, batch_size):
            new_mat[i][j] = torch.dot(mat[i], mat[j])

    # new_mat will be an upper triangular matrix
    i_lower = np.tril_indices(batch_size)
    new_mat[i_lower] = new_mat.transpose(0, 1)[i_lower]

    return new_mat


def generate_featuremaps(net, image_data, targets, args, fmap, isStudent=True):
    """
    Generate gradient matrix of size (batch-size x num_params)
    """
    batch_size = image_data.shape[0]
    # Assume that only the last conv layer is used for calculation
    # Student is resnet-32 and teacher is resnet-56
    if isStudent:
        last_conv = net.layer3[4].conv2
    else:
        last_conv = net.layer3[8].conv2

    net.eval()

    for index, (image, target) in enumerate(zip(image_data, targets)):
        # unsqueeze adds the extra batch-size dimension
        logits = net(image.unsqueeze(0), is_feat=False, preact=False)
        fmap[index] = torch.autograd.grad(logits[0, target], last_conv.parameters(), retain_graph=True,
                                          create_graph=True)[0].flatten()

    if isStudent:
        net.train()

    return fmap