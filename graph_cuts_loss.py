
# -*- coding: utf-8 -*-
import torch
import torch.nn


# original 2D GC loss with no approximation
class GC_2D_Original(torch.nn.Module):

    def __init__(self, lmda, sigma):
        super(GC_2D_Original, self).__init__()
        self.lmda = lmda
        self.sigma = sigma

    def forward(self, input, target):
        # input: B * C * H * W, after sigmoid operation
        # target: B * C * H * W
