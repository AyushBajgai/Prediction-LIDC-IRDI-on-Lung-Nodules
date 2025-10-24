import argparse


def str2bool(v):
    # Convert string input to boolean for argparse
    v = v.lower()
    if v in ['true', '1']:
        return True
    elif v in ['false', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (true/false or 1/0).')


def count_params(model):
    # Return total number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Compute and store the current value and running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Reset all statistics
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # Update with a new value
        # val: the current value
        # n: number of samples used to calculate average
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
