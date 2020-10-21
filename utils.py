import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


class LrReducer():
    def __init__(self, tot_epochs, decay_epoch):
        self.tot_epochs = tot_epochs 
        self.decay_epoch = decay_epoch 
        self.curr_ratio = 1 
        self.halving_epoch = int(self.tot_epochs * 0.2) 

    def step(self, curr_epoch):
        if curr_epoch > self.decay_epoch and curr_epoch % self.halving_epoch == 0:
            self.curr_ratio = self.curr_ratio / 2
        return self.curr_ratio 


def weights_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.bias.data, 0.0)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('LeakageLayer') != -1:
        nn.init.constant_(m.weight.data, 1)


class MSImWarp():
    def __init__(self, device=torch.device('cuda'), scales=6, im_res=[256, 512], batch_size=16):
        self.scales = scales 
        self.im_res = im_res 
        self.batch_size = batch_size 
        self.device = device 

        # new image resolutions 
        self.res = []
        curr_scale = 2
        for _ in range(scales):
            self.res += [tuple(np.array(self.im_res) // curr_scale)]
            curr_scale *= 2
        # print(self.res)
        # making the meshgrids required
        self.grids = []
        for h, w in self.res:
            h = int(h)
            w = int(w)
            dh = torch.linspace(-1, 1, h)
            dw = torch.linspace(-1, 1, w)
            meshx, meshy = torch.meshgrid((dh, dw))
            grid = torch.stack((meshy, meshx), 2)
            grid = grid.unsqueeze(0)
            grid_batch = []
            for _ in range(self.batch_size):
                grid_batch += [grid] 
            grid_batch = torch.cat(grid_batch, dim=0).to(self.device)
            self.grids += [grid_batch]
        
    def __call__(self, images, dmaps, im_warping=True):
        assert len(images) == len(dmaps) or len(images) == 1, \
            'images: {}, dmaps: {}'.format(len(images), len(dmaps))
        
        # resizing images if required
        res_images = []
        if len(images) == 1:
            for i, _ in enumerate(range(self.scales)): 
                new_im = F.interpolate(images, self.res[i], mode='bilinear', align_corners=False)
                res_images += [new_im]
        elif im_warping:
            for image in images:
                im = image.clone()
                res_images += [im]
        else:
            for image in images:
                im = image.clone()
                res_images += [im]

        assert len(res_images) == self.scales, 'Images: {} vs Scales: {}'.format(len(res_images),
                                                                                 self.scales)
        assert len(res_images) == len(dmaps), 'Images: {} vs dmaps: {}'.format(len(res_images), 
                                                                               len(dmaps))

        # applying image warping
        recon_ims = []
        i = 0
        for im, dmap in zip(res_images, dmaps):
            im_res = [im.size(-2), im.size(-1)]
            dmap_res = [dmap.size(-2), dmap.size(-1)]
            assert im_res == dmap_res, 'image, dmap size mismatch {} vs {}'.format(im.size(),
                                                                                         dmap.size())
            recon_ims += [self.imwarp(im, dmap, i)] 
            i += 1
        del dmap, res_images
        return recon_ims 


    def imwarp(self, image, dmap, i):
        w = dmap.size(-1)
        dmap_ = dmap.squeeze(1)
        grid_ = self.grids[i].clone()
        grid_[:, :, :, 0] -= dmap_ * 2.0 / w 
        mapped_out = F.grid_sample(image, grid_, mode='bilinear', align_corners=False)
        return mapped_out


def grads(image):
    if len(image.size()) == 3:
        image = image.unsqueeze(1)
    assert len(image.size()) == 4, '4 d input expected, got {}'.format(image.size())
    left_im = F.pad(image, (0, 1), 'constant', 0)
    dx = image - left_im[:, :, :, 1:]
    up_im = F.pad(image, (0, 0, 0, 1), 'constant', 0)
    dy = image - up_im[:, :, 1:, :]
    dx = torch.mean(dx, dim=1).unsqueeze(1)
    dy = torch.mean(dy, dim=1).unsqueeze(1)
    return dx, dy


class SSIM():
    def __init__(self, win_size=3, nc=3):
        self.win_size = win_size 
        self.nc = nc
        self.kernel = 1.0 / win_size / win_size * \
             torch.ones((1, self.nc, win_size, win_size))
        self.padding_size = self.win_size // 2
        self.c1 = 0.01 * 0.01 
        self.c2 = 0.03 * 0.03 

    def __call__(self, im1, im2):
        assert im1.size() == im2.size(), 'Image size mismatch. {} with {}'.format(im1.size(), im2.size())
        self.kernel = self.kernel.to(im1.device)
        mean1 = F.conv2d(im1, self.kernel, padding=self.padding_size)
        mean2 = F.conv2d(im2, self.kernel, padding=self.padding_size)
        mean1_mean2 = mean1 * mean2 
        mean1_sq = mean1 * mean1 
        mean2_sq = mean2 * mean2 

        var1 = F.conv2d(im1 * im1, self.kernel, padding=self.padding_size) - mean1_sq 
        var2 = F.conv2d(im2 * im2, self.kernel, padding=self.padding_size) - mean2_sq 
        covar12 = F.conv2d(im1 * im2, self.kernel, padding=self.padding_size) - mean1_mean2

        ssim_im = ((2.0 * mean1_mean2 + self.c1) * (2.0 * covar12 + self.c2)) / \
                  ((mean1_sq + mean2_sq + self.c1) * (var1 + var2 + self.c2))
        ssim_loss = torch.mean(ssim_im)

        return ssim_loss.clamp(0.0, 1.0)
