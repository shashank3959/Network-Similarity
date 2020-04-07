from __future__ import print_function

import torch
import numpy as np
from functools import reduce

def compute_rowwise_dot(mat):
    """
    Computes row-row dot products of matrix. Resultant matrix of size batch-size x batch-size.
    """
    batch_size = mat.shape[0]
    new_mat = torch.zeros(batch_size, batch_size)
    if torch.cuda.is_available():
        new_mat = new_mat.cuda()
    # Calculate only the upper triangular part to save on half calculations
    for i in range(batch_size):
        for j in range(i, batch_size):
            new_mat[i][j] = torch.dot(mat[i], mat[j])

    # new_mat will be an upper triangular matrix
    i_lower = np.tril_indices(batch_size)
    new_mat[i_lower] = new_mat.transpose(0, 1)[i_lower]
    return new_mat


def generate_featuremaps(net, image_data, targets, args, isStudent=True):
    """
    Generate NTK feature maps of size batch-size x batch-size
    """
    batch_size = image_data.shape[0]
    # Assume that only the last conv layer is used for calculation
    # Student is resnet-32 and teacher is resnet-56
    if isStudent:
        last_conv = net.layer3[4].conv2
        layer_params = net.state_dict()['layer3.4.conv2.weight']
    else:
        last_conv = net.layer3[8].conv2
        layer_params = net.state_dict()['layer3.8.conv2.weight']

    num_params = reduce(lambda x, y: x * y, layer_params.shape)

    fmap = torch.zeros([batch_size, num_params], requires_grad=True)
    if torch.cuda.is_available():
        fmap = fmap.cuda()

    net.eval()

    # This loop will run batch-size times
    for index, (image, target) in enumerate(zip(image_data, targets)):
        # unsqueeze adds the extra batch-size dimension
        logits = net(image.unsqueeze(0), is_feat=False, preact=False)
        fmap[index] = torch.autograd.grad(logits[0, target], last_conv.parameters(), retain_graph=True,
                                          create_graph=True)[0].flatten()

    if isStudent:
        net.train()

    return compute_rowwise_dot(fmap)