import torch 
import torch.utils.data as data 
import torchvision.transforms as transforms 
import torchvision.utils as vutils
import torch.nn as nn 
import numpy as np 
import argparse
import importlib
import matplotlib as mpl
from dataset import StereoLoader 


class Test():
    def __init__(self, input_dir='./opa_dataset/', output_dir='./output_images/', module=None, gpus=[1],
                 im_size=[256, 512], model='./models/0000'):
        self.input_dir = input_dir 
        self.output_dir = output_dir 
        self.module = module 
        self.gpus = gpus 
        self.im_size = im_size 
        self.model = model

        # device type 
        if len(self.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(self.gpus[0]))
        
        # dataloading
        transform = [transforms.Resize(self.im_size),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        Dataset = StereoLoader(root=self.input_dir, req_transforms=transform)
        self.DataLoader = data.DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=4)

        # loading the network 
        self.DispGen = self.module.DisparityGen(in_channels=3, batch_size=1).to(self.device)
        self.DispGen.load_state_dict(torch.load(self.model))
        if len(self.gpus) > 0:
            self.DispGen = nn.DataParallel(self.DispGen, device_ids=self.gpus)
        
        # launching testing 
        self.testing()

    def testing(self):
        with torch.no_grad():
            for i, data in enumerate(self.DataLoader):
                left_im = data['left'].to(self.device)
                right_im = data['right'].to(self.device)
                out_data = self.DispGen(left_im, right_im)
                self.save_result(i, left_im, right_im, out_data)

                if i % 10 == 0:
                    print('Images done {} of {}'.format(i, len(self.DataLoader)))
        print('Testing done')
            
    def save_result(self, curr_iter, fixed_left, fixed_right, res_dict):
        iter_str = ('%07d' % curr_iter)
        im_name = self.output_dir + iter_str + '.png'
        # with torch.no_grad():
        #    res_dict = self.Net(self.fixed_right, self.fixed_left)
        
        im_rec_r = (res_dict['im_rec_r'])[0]
        disp_r = (res_dict['disp_r'])[0]
        # disp_r = self.colormap(disp_r)
        disp_r = disp_r.repeat(3, 1, 1)
        im_r = fixed_right[0]
        im_l = fixed_left[0]
        im_rec_r = vutils.make_grid(im_rec_r, normalize=True, padding=0)
        disp_r = vutils.make_grid(disp_r, normalize=True, padding=0)
        im_l = vutils.make_grid(im_l, normalize=True, padding=0)
        im_r = vutils.make_grid(im_r, normalize=True, padding=0)
        hor_im1 = torch.cat((im_r, im_rec_r), dim=-1)
        hor_im2 = torch.cat((im_l, disp_r), dim=-1)
        full_im = torch.cat((hor_im1, hor_im2), dim=-2)
        vutils.save_image(full_im, im_name)

    def colormap(self, disp_in):
        cm_new = mpl.cm.get_cmap('jet')
        disp = disp_in.cpu().numpy()
        b, h, w = disp.shape 
        disp_new = np.zeros((b, h, w, 3))
        for sample in range(b):
            disp_new[sample, :, :, :] = cm_new(255 * disp[sample, :, :])[:, :, :3]
        disp_new = np.transpose(disp_new, (0, 3, 1, 2))
        disp_new = torch.tensor(disp_new, dtype=disp_in.dtype).to(self.device)[:, :3, :, :] 
        # print(disp_new.size())
        return disp_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default='/hdd/local/sdb/umar/kitti_dataset2/')
    parser.add_argument('--output_dir', type=str, default='./output_results/')
    parser.add_argument('--network_file', type=str, default='network3')
    parser.add_argument('--gpus', type=list, default=[1])
    parser.add_argument('--model', type=str, default='./models/000_0010000.pth')
    
    args = parser.parse_args()

    module = importlib.import_module(args.network_file)
    input_dir = args.input_dir 
    output_dir = args.output_dir 
    gpus = args.gpus 
    model = args.model 

    Test(input_dir=input_dir, 
         output_dir=output_dir, 
         module=module, 
         gpus=gpus,
         model=model)