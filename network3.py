import torch
import torch.nn as nn
import torch.nn.functional as F 


class LeakageLayer(nn.Module):
    def __init__(self, in_channels=3, in_size=[256, 512]):
        self.in_channels = in_channels 
        self.in_size = in_size
        h, w = self.in_size[0], self.in_size[1] 

        self.mask = nn.Parameter(torch.Tensor(1, self.in_channels, h, w))

    def forward(self, x):
        return x * self.mask

class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super(DownSampler, self).__init__()
        self.in_c = in_channels 
        self.out_c = out_channels 
        self.k_size = k_size
        self.pad_val = k_size // 2
        
        self.ops = []
        self.ops += [nn.ReflectionPad2d(self.pad_val)]
        self.ops += [nn.Conv2d(self.in_c, self.out_c, self.k_size, 2)]
        self.ops += [nn.LeakyReLU(0.2, inplace=True)]
        self.ops += [nn.ReflectionPad2d(self.pad_val)]
        self.ops += [nn.Conv2d(self.out_c, self.out_c, self.k_size, 1)]
        self.ops += [nn.ReLU()]
        self.ops += [nn.LeakyReLU(0.2, inplace=True)]
        self.model = nn.Sequential(*self.ops)

    def forward(self, x):
        return self.model(x)

    
class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, skip=True):
        super(Upsampler, self).__init__()
        self.in_c = in_channels 
        self.out_c = out_channels 
        self.k_size = 3
        self.skip = skip
        
        self.ops1 = []
        # self.ops += [nn.ReflectionPad2d(1)]
        self.ops1 += [nn.ConvTranspose2d(self.in_c, self.out_c, 3, 2, padding=1, output_padding=1)]
        self.ops1 += [nn.LeakyReLU(0.2, inplace=True)]
        self.model1 = nn.Sequential(*self.ops1)
        self.ops2 = []
        self.ops2 += [nn.ReflectionPad2d(1)]
        if self.skip:
            self.ops2 += [nn.Conv2d(self.out_c * 2, self.out_c, 3, 1)]        
        else: 
            self.ops2 += [nn.Conv2d(self.out_c, self.out_c, 3, 1)]
        self.ops2 += [nn.LeakyReLU(0.2, inplace=True)]
        self.model2 = nn.Sequential(*self.ops2)

    def forward(self, x, x_skip=None):
        y1 = self.model1(x)
        if self.skip:
            cat_feature = torch.cat((y1, x_skip), dim=1)
        else:
            cat_feature = y1
        return self.model2(cat_feature)


class DisparityGen(nn.Module):
    def __init__(self, in_channels=3, batch_size=4):
        super(DisparityGen, self).__init__()
        self.batch_size = batch_size
        self.in_c = in_channels  
        
        # Encoder 
        self.enc_channels_list = [32, 64, 128, 256, 512, 512, 512]
        self.k_size_list = [7, 5, 3, 3, 3, 3, 3]
        # self.out_res_min = [128, 64, 32, 16, 8, 4, 2]
        self.enc_ops = nn.ModuleList()
        in_c = self.in_c 
        for out_c, k_size in zip(self.enc_channels_list, self.k_size_list):
            self.enc_ops += [DownSampler(in_c, out_c, k_size)]
            in_c = out_c
        
        # decoder 
        self.dec_channels_list = [512, 512, 256, 128, 64, 32]
        # self.out_res_min = [4, 8, 16, 32, 64, 128, 256]
        self.dec_ops = nn.ModuleList()
        for out_c in self.dec_channels_list:
            self.dec_ops += [Upsampler(in_c, out_c)]
            in_c = out_c 
        
        # disp_op
        self.disp_op = []
        self.disp_op += [Upsampler(32, 16, skip=False)]
        self.disp_op += [nn.ReflectionPad2d(1)]
        self.disp_op += [nn.Conv2d(16, 1, 3, 1)]
        self.disp_op += [nn.Sigmoid()]
        self.disp_model = nn.Sequential(*self.disp_op)
        self.ImWarper = ImWarper2()

    def forward(self, im_r, im_l):
        # Encoding and encoder features 
        enc_feat = []
        feat = im_l
        for i in range(len(self.enc_channels_list)):
            feat = self.enc_ops[i](feat)
            enc_feat += [feat]
            
        enc_feat.reverse()
        for i in range(len(self.dec_channels_list)):
            feat = self.dec_ops[i](feat, enc_feat[i + 1])

        disp_r = 0.3 * self.disp_model(feat)
        # disp_l = disp[:, 0, :, :].unsqueeze(1)
        # disp_r = disp[:, 1, :, :].unsqueeze(1)
        im_rec_r = self.ImWarper(im_l, disp_r)
        # im_rec_l = self.ImWarper(im_r, -1.0 * disp_l)
        return {'disp_r': disp_r, 'im_rec_r': im_rec_r}


class ImWarper(nn.Module):
    def __init__(self, res=[256, 512], batch_size=16):
        super(ImWarper, self).__init__()
        self.res = res
        self.batch_size = batch_size
        self.device = torch.device('cuda')
        h = self.res[0]
        w = self.res[1]
        dh = torch.linspace(-1, 1, h)
        dw = torch.linspace(-1, 1, w)
        meshx, meshy = torch.meshgrid((dh, dw))
        print(meshx)
        print(meshy)
        print(meshx.size())
        print(meshy.size())
        grid = torch.stack((meshy, meshx), 2)
        print(grid.size())
        self.grid = grid.unsqueeze(0)
        self.grid_batch = self.grid.repeat(self.batch_size, 1, 1, 1)
        print(self.grid_batch.size())

    def forward(self, dmap, image):
        im_h, im_w = image.size(-2), image.size(-1)
        dmap_h, dmap_w = dmap.size(-2), dmap.size(-1)
        assert (im_h, im_w) == (dmap_h, dmap_w), 'Disparity, image dims mismatch'

        if len(image.size()) < 4: # disparity map with no channels 
            image = image.unsqueeze(1)
        
        # dmap_ = dmap.squeeze(1)
        # grid_ = grid_batch.clone()
        grid_batch = self.grid_batch.clone().to(dmap.device)
        grid_batch[:, :, :, 0] += (2.0 * dmap - 1.0)
        mapped_out = F.grid_sample(image, grid_batch, mode='bilinear', align_corners=False)
        return mapped_out


class ImWarper2(nn.Module):
    def __init__(self, res=[256, 512], device=torch.device('cuda')):
        super(ImWarper2, self).__init__()
        self.res = res

    def forward(self, img, disp):
        im_resized = F.interpolate(img, self.res, mode='bilinear', align_corners=False)
        img=im_resized
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        # x_shifts = disp
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2.0 * flow_field - 1.0, mode='bilinear',
                               padding_mode='zeros', align_corners=False)
        return output
