import torchaudio.transforms as T
from torch import nn, Tensor
from src.transforms.random_apply import RandomApply

class FrequencyMasking(nn.Module):
    def __init__(self, p, *args, **kwargs):
        super().__init__()
        self._aug = T.FrequencyMasking(*args, **kwargs)
        self.p = p
    
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return RandomApply(self._aug)(x).squeeze(1)


class TimeMasking(nn.Module):
    def __init__(self, p, *args, **kwargs):
        super().__init__()
        self._aug = T.TimeMasking(*args, **kwargs)
        self.p = p

    
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return RandomApply(self._aug)(x).squeeze(1)
    

class TimeStretch(nn.Module):
    def __init__(self, p, *args, **kwargs):
        super().__init__()
        self._aug = T.TimeStretch(*args, **kwargs)
        self.p = p

    
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return RandomApply(self._aug)(x).squeeze(1)
