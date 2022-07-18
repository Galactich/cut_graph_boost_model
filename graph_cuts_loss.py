
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

        return self.lmda * region_term + boundary_term


# 2D GC loss with boundary approximation in equation (7) to eliminate sigma
class GC_2D(torch.nn.Module):

    def __init__(self, lmda):
        super(GC_2D, self).__init__()
        self.lmda = lmda

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
        input_vert = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])  # |pu - pv|

        # horizontal: x <-> x2, x4 <-> x
        target_hori = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])  # delta(yu, yv)
        input_hori = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])  # |pu - pv|

        # diagonal1: x <-> x5, x8 <-> x
        target_diag1 = torch.abs(target[:, :, 1:, 1:] - target[:, :, :-1, :-1])  # delta(yu, yv)
        input_diag1 = torch.abs(input[:, :, 1:, 1:] - input[:, :, :-1, :-1])  # |pu - pv|

        # diagonal2: x <-> x7, x6 <-> x
        target_diag2 = torch.abs(target[:, :, 1:, :-1] - target[:, :, :-1, 1:])  # delta(yu, yv)
        input_diag2 = torch.abs(input[:, :, 1:, :-1] - input[:, :, :-1, 1:])  # |pu - pv|

        p1 = input_vert * target_vert
        p2 = input_hori * target_hori
        p3 = input_diag1 * target_diag1
        p4 = input_diag2 * target_diag2

        boundary_term = 1 - (torch.sum(p1) / torch.sum(target_vert) +
                             torch.sum(p2) / torch.sum(target_hori) +
                             torch.sum(p3) / torch.sum(target_diag1) +
                             torch.sum(p4) / torch.sum(target_diag2)) / 4  # equation (7), and normalized to (0,1)

        return self.lmda * region_term + boundary_term


# 3D GC loss with boundary approximation in equation (7) to eliminate sigma
class GC_3D_v1(torch.nn.Module):
    def __init__(self, lmda):
        super(GC_3D_v1, self).__init__()
        self.lmda = lmda

    def forward(self, input, target):
        # input: B * C * H * W * D, after sigmoid operation
        # target: B * C * H * W * D

        # region term
        bce = torch.nn.BCELoss()
        region_term = bce(input=input, target=target)

        # boundary term
        '''
        example [[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]],[[19, 20, 21], [22, 23, 24], [25, 26, 27]]]]]
        element 14 has 26 neighborhoods, a total of 13 operations
        '''
        # x5 <-> x14, x14 <-> x23
        input_1 = torch.abs(input[..., 1:, :, :] - input[..., :-1, :, :])  # |pu - pv|
        target_1 = torch.abs(target[..., 1:, :, :] - target[..., :-1, :, :])  # delta(yu, yv)
        # x11 <-> x14, x14 <-> x17
        input_2 = torch.abs(input[..., :, 1:, :] - input[..., :, :-1, :])
        target_2 = torch.abs(target[..., :, 1:, :] - target[..., :, :-1, :])
        # x13 <-> x14, x14 <-> x15
        input_3 = torch.abs(input[..., :, :, 1:] - input[..., :, :, :-1])
        target_3 = torch.abs(target[..., :, :, 1:] - target[..., :, :, :-1])
        # x2 <-> x14, x14 <-> x26
        input_4 = torch.abs(input[..., 1:, 1:, :] - input[..., :-1, :-1, :])
        target_4 = torch.abs(target[..., 1:, 1:, :] - target[..., :-1, :-1, :])
        # x8 <-> x14, x14 <-> x20
        input_5 = torch.abs(input[..., 1:, :-1, :] - input[..., :-1, 1:, :])