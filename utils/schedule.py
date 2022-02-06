import torch

class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    # Reference : https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/optimization.py#L33
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1.0, warmup_epochs))
            return 1.

        super(WarmupConstantSchedule, self).__init__(
            optimizer, lr_lambda, last_epoch=last_epoch)
