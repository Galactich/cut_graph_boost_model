
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

        # region term equals to BCE
        bce = torch.nn.BCELoss()
        region_term = bce(input=input, target=target)

        # boundary_term
        '''
        x5 x1 x6
        x2 x  x4
        x7 x3 x8
        '''
        # vertical: x <-> x1, x3 <-> x1
        target_vert = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])  # delta(yu, yv)
        input_vert = input[:, :, 1:, :] - input[:, :, :-1, :]  # pu - pv
