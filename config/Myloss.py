import torch
from torch import nn
from torch.nn import L1Loss, MSELoss
from DarkChannel import DarkChannel


class Myloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = L1Loss()
        self.L2 = MSELoss()
        self.dark = DarkChannel()

    def forward(self, x, y):
        a = (self.L2(self.dark(x), self.dark(y))).to(torch.float32).cuda()
        b = self.L1(x, y)
        return a + b
