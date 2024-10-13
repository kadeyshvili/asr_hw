from torch import Tensor, nn
from torchaudio.transforms import SpeedPerturbation


class MySpeedPerturbation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = SpeedPerturbation(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x)[0].squeeze(1)