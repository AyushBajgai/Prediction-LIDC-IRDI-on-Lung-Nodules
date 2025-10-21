import argparse

def str2bool(v: str) -> bool:
    """Convert string to boolean (for argparse)."""
    v = str(v).lower()
    if v in ('true', '1', 'yes', 'y'):
        return True
    elif v in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")

def count_trainable_params(model):
    """Return number of trainable parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

class AverageMeter:
    """Compute and store the running average of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.total = 0.0
        self.count = 0
        self.avg = 0.0
        self.sum = 0.0

    def update(self, val, n=1):
        self.value = val
        self.total += val * n
        self.count += n
        self.avg = self.total / self.count if self.count != 0 else 0.0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
