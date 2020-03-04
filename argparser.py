import argparse


def argparser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--show_images', action='store_true', help='Display a random image')

    # Training Hyperparameters
#     parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--num_workers', default=2, type=float, help='number of workers')

    # Model State arguments
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

    args = parser.parse_args()

    return args