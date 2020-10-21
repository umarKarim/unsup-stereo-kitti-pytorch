import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from utils import MSImWarp, grads, SSIM
import numpy as np


class Losses():
    def __init__(self, weights=[1.0, 0.0, 0.0], alpha=0.85):
        self.weights = weights

        self.alpha = alpha 
        self.ApLoss = AppearanceLoss(self.alpha)
        self.SmLoss = SmoothnessLoss()
        self.ConsistLoss = LRConsistency()

        # defining the new sizes
        self.ap_loss = 0
        self.sm_loss = 0
        self.consist_loss = 0

    def __call__(self, im_r, net_dict):
        im_rec_r = net_dict['im_rec_r']
        self.ap_loss = self.ApLoss(im_r, im_rec_r)
        return self.ap_loss


class AppearanceLoss():
    def __init__(self, alpha=0):
        self.alpha = 0.85
        self.L2Loss = nn.L1Loss()
        self.SSIMLoss = SSIM(nc=3)

    def SSIM_loss(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def __call__(self, im1, im2):
        loss = self.alpha * (self.SSIM_loss(im1, im2))  + (1 - self.alpha) * \
                self.L2Loss(im2, im1)
        return torch.mean(loss)

class SmoothnessLoss():
    def __init__(self):
        pass 

    def __call__(self, dmap, im):
        grad_x_im, grad_y_im = grads(im)
        grad_x_d, grad_y_d = grads(dmap)
        
        grad_x_im, grad_y_im = grad_x_im[:, :, 0:-1, 0:-1], grad_y_im[:, :, 0:-1, 0:-1]
        grad_x_d, grad_y_d = grad_x_d[:, :, 0:-1, 0:-1], grad_y_d[:, :, 0:-1, 0:-1]

        wt_x = torch.exp(-1.0 * torch.abs(grad_x_im))
        wt_y = torch.exp(-1.0 * torch.abs(grad_y_im))

        sm_x = torch.abs(grad_x_d * wt_x) 
        sm_y = torch.abs(grad_y_d * wt_y) 

        loss = torch.mean(sm_x) + torch.mean(sm_y)
        return loss
        

class LRConsistency():
    def __init__(self):
        self.L1Loss = nn.L1Loss()

    def __call__(self, dmaps1, dmaps2):
        return torch.mean(self.L1Loss(dmaps1, dmaps2))

