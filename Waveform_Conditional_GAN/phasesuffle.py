import random
import torch
import torch.nn as nn

def PhaseShuffle(x, rad):
    batch_size = x.size(0)
    channel = x.size(1)
    length = x.size(2)

    phase = random.randrange(-rad, rad+1)
    pad_l = max(phase, 0)
    pad_r = max(-phase, 0)
    phase_start = pad_r

    x = nn.ReflectionPad1d((pad_l, pad_r))(x)
    x = x[:, :, phase_start:phase_start+length]
    x = x.view(batch_size, channel, length)

    return x