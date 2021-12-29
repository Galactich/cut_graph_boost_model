
# -*- coding: utf-8 -*-
import torch
import torch.nn


# original 2D GC loss with no approximation
class GC_2D_Original(torch.nn.Module):

    def __init__(self, lmda, sigma):
        super(GC_2D_Original, self).__init__()
        self.lmda = lmda