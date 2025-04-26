from .cosine_lr import CosineLRScheduler

def create_scheduler(optimizer, epochs, lr):

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epochs,
            lr_min=1e-6,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=1e-5,
            warmup_t=5,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler


from bisect import bisect_right
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

'''class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,  # steps
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=20,
            warmup_method="linear",
            last_epoch=-1,
    ):
        
        :param optimizer: 优化器
        :param milestones: 在第几个epoch降低学习率
        :param gamma: 降低gamma倍学习率
        :param warmup_factor:预热因子，即预热期间学习率相对于初始学习率的比例，默认为 1/3。
        :param warmup_iters:warmup 期间的 epoch 数量。
        :param warmup_method:warmup 策略。可以是 "constant" 或 "linear"。
        "constant" 表示在前几个 epoch 中保持学习率不变，"linear" 表示逐渐增加学习率。
        :param last_epoch:在初始化时，last_epoch 的初始值为 -1，表示还没有进行任何训练。
        随着每个 epoch 的结束，last_epoch 的值会逐渐增加，以便调度器可以根据当前的 epoch 来调整学习率。
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]'''

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,  # steps
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]