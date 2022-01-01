
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

        # horizontal: x <-> x2, x4 <-> x
        target_hori = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])  # delta(yu, yv)
        input_hori = input[:, :, :, 1:] - input[:, :, :, :-1]  # pu - pv

        # diagonal1: x <-> x5, x8 <-> x
        target_diag1 = torch.abs(target[:, :, 1:, 1:] - target[:, :, :-1, :-1])  # delta(yu, yv)
        input_diag1 = input[:, :, 1:, 1:] - input[:, :, :-1, :-1]  # pu - pv

        # diagonal2: x <-> x7, x6 <-> x
        target_diag2 = torch.abs(target[:, :, 1:, :-1] - target[:, :, :-1, 1:])  # delta(yu, yv)
        input_diag2 = input[:, :, 1:, :-1] - input[:, :, :-1, 1:]  # pu - pv

        dist1 = 1.0  # dist(u, v), e.g. x <-> x1
        dist2 = 2.0 ** 0.5  # dist(u, v) , e.g. x <-> x6

        p1 = torch.exp(-(input_vert ** 2) / (2 * self.sigma * self.sigma)) / dist1 * target_vert
        p2 = torch.exp(-(input_hori ** 2) / (2 * self.sigma * self.sigma)) / dist1 * target_hori

        p3 = torch.exp(-(input_diag1 ** 2) / (2 * self.sigma * self.sigma)) / dist2 * target_diag1
        p4 = torch.exp(-(input_diag2 ** 2) / (2 * self.sigma * self.sigma)) / dist2 * target_diag2

        boundary_term = (torch.sum(p1) / torch.sum(target_vert) +
                         torch.sum(p2) / torch.sum(target_hori) +
                         torch.sum(p3) / torch.sum(target_diag1) +
                         torch.sum(p4) / torch.sum(target_diag2)) / 4  # equation (5)