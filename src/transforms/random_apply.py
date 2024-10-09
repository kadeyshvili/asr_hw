import random
from torch import Tensor
from typing import Callable


class RandomApply:
    def __init__(self, augmentation: Callable):
        self.augmentation = augmentation[0]._aug
        self.p = self.augmentation.p
        assert 0 <= self.p <= 1


    def __call__(self, data: Tensor) -> Tensor:
        if random.random() < self.p:
            return self.augmentation(data)
        else:
            return data