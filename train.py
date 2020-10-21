import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
import torchvision.utils as vutils
import numpy as np 
import torch.utils.tensorboard.writer as Writer  
import torch.utils.data as data 
import argparse
import importlib 
from dataset import StereoLoader 
import matplotlib.pyplot as plt 
import os 
from utils import weights_init, LrReducer 
import time 
import sys 


class Trainer():
    def __init__(self, epochs=50, batch_size=16, gpus=[0], net_module=None, losses_module=None, 
                 lr=0.0001, beta1=0.5, beta2=0.99, root='/hdd/local/sdb/umar/kitti_dataset2/',
                 image_size=[256, 512], model_path='./models/', summary_path='./summary/', 
                 load_pretrained=True, save_model_iter=10000, save_result_iter=100, 
                 res_dir='./curr_res/'):
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.gpus = gpus 
        self.net_module = net_module 
        self.losses_module = losses_module 
        self.lr = lr 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.root = root 
        self.image_size = image_size 
        self.model_path = model_path 
        self.summary_path = summary_path 
        self.load_pretrained = load_pretrained
        # self.writer = Writer.SummaryWriter(log_dir=self.summary_path)
        self.save_model_iter = save_model_iter 
        self.save_result_iter = save_result_iter
        self.res_dir = res_dir 

        # self.decay_epoch = 0.6 * self.epochs 
        self.decay_epoch = 30
        self.st_time = time.time()         

        # data loader 
        req_transforms = [transforms.Resize(self.image_size),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = StereoLoader(root=self.root, req_transforms=req_transforms)
        print('Total examples: {}'.format(len(dataset)))
        self.data_loader = data.DataLoader(dataset, batch_size=self.batch_size,
                                           shuffle=True, num_workers=8, pin_memory=True,
                                           drop_last=True)
        # self.sample_display()

        # setting the device 
        if len(self.gpus) > 0:
            self.device = torch.device('cuda:0')
        else: 
            self.device = torch.device('cpu')
        print('The device type is: {}'.format(self.device))
        print('GPUs used: {}'.format(self.gpus))
        
        # loading the models 
        self.Net = net_module.DisparityGen( 
                                          in_channels=3, batch_size=self.batch_size).to(self.device)
        
        if self.load_pretrained:
            saved_models = os.listdir(self.model_path).sort()
            # print(saved_models)
            if saved_models is not None :
                saved_models = [x for x in saved_models if x.endswith('.pth')]

        if saved_models is None or not self.load_pretrained:
            pass
            # self.Net.apply(weights_init)
        else:
            model_name = self.model_path + saved_models[-1]
            self.Net.load_state_dict(torch.load(model_name))
        if self.device.type == 'cuda':
            self.Net = nn.DataParallel(self.Net, device_ids=self.gpus)
        
        # loss function
        self.criterion_Loss = self.losses_module.Losses()

        # optimizer 
        self.optim = torch.optim.Adam(self.Net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2),
                                      eps=10e-8)
        self.LrReducer = LrReducer(self.epochs, self.decay_epoch)
        self.LrScheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=self.LrReducer.step)
        
        # writing outputs for tensorboard 
        print('Summaries written to: {}'.format(self.summary_path))
        
        # fixed images for intermediat results checking
        fixed_data = next(iter(self.data_loader))
        self.fixed_right = fixed_data['right'].to(self.device)
        self.fixed_left = fixed_data['left'].to(self.device)

        # training begins 
        self.train()

    def train(self):
        for epoch in range(self.epochs):
            for i, data in enumerate(self.data_loader):
                self.optim.zero_grad()
                left_im, right_im = data['left'].to(self.device), data['right'].to(self.device)
                net_dict = self.Net(right_im, left_im)
                # net_dict = self.Net(self.fixed_right, self.fixed_left)

                loss = self.criterion_Loss(right_im, net_dict)
                loss.backward()
                self.optim.step()
                # print('time for updating the net: {}'.format(time.time() - curr_time))
                
                global_step = epoch * len(self.data_loader) + i
                if i % 50 == 0: 
                    template = """ Epoch: {}/{}, iteration: {}/{}, time: {}, train loss: {}, 
                                   app loss: {}, sm loss: {}, lr loss: {}""" 
                    print(template.format(epoch, self.epochs, i, len(self.data_loader), 
                                          time.time() - self.st_time, loss.item(),
                                          self.criterion_Loss.ap_loss.item(),
                                          0, 0))
                    max_disp = torch.max(net_dict['disp_r'])
                    min_disp = torch.min(net_dict['disp_r'])
                    mean_disp = torch.mean(net_dict['disp_r'])
                    print('Max: {}, min: {} and mean: {} disparities'.format(max_disp, min_disp, mean_disp))
                    '''self.writer.add_scalars('train_loss', {'app_loss' : self.criterion_Loss.ap_loss.item(),
                                                           'sm_loss' : self.criterion_Loss.sm_loss.item(),
                                                           'lr_loss' : self.criterion_Loss.consist_loss.item()},
                                            global_step=global_step)'''

                if global_step % self.save_result_iter == 0:
                    # self.write_2_tboard(left_im, right_im, net_dict, global_step)
                    self.save_result(i, epoch)
                    print('Image saved')
                    
                if global_step % self.save_model_iter == 0:
                    self.save_model(epoch, i)

                if global_step % 200 == 0:
                    torch.cuda.empty_cache()

    def save_result(self, curr_iter, epoch):
        epoch_str = ('%03d_' % epoch)
        iter_str = ('%07d' % curr_iter)
        im_name = self.res_dir + epoch_str + iter_str + '.png'
        with torch.no_grad():
            res_dict = self.Net(self.fixed_right, self.fixed_left)
        im_rec_r = (res_dict['im_rec_r'])[0]
        disp_r = (res_dict['disp_r'])[0]
        # disp_l = torch.cat((disp_l, disp_l, disp_l), dim=0)
        disp_r = disp_r.repeat(3, 1, 1)
        im_r = self.fixed_right[0]
        im_l = self.fixed_left[0]
        # disp_r = torch.cat((disp_r, disp_r, disp_r), dim=0)
        # disp_r = disp_r.repeat(3, 1, 1)
        im_rec_r = vutils.make_grid(im_rec_r, normalize=True, padding=0)
        disp_r = vutils.make_grid(disp_r, normalize=True, padding=0)
        im_l = vutils.make_grid(im_l, normalize=True, padding=0)
        im_r = vutils.make_grid(im_r, normalize=True, padding=0)
        # disp_r = vutils.make_grid(disp_l, normalize=True, padding=0)
        
        hor_im1 = torch.cat((im_r, im_rec_r), dim=-1)
        hor_im2 = torch.cat((im_l, disp_r), dim=-1)
        full_im = torch.cat((hor_im1, hor_im2), dim=-2)
        # full_im = torch.cat((im_l, im_rec_l, disp_l), dim=-1)
        vutils.save_image(full_im, im_name)
    
    def save_model(self, epoch, i):
        epoch_str = ('%03d_' % epoch)
        iter_str = ('%07d' % i)
        model_name = self.model_path + epoch_str + iter_str + '.pth'
        torch.save(self.Net.module.state_dict(), model_name)
        print('Saved model')
        # torch.save(self.disB.module.state_dict(), self.model_path + 'DisB_' + epoch_str + '.pth')

    def write_2_tboard(self, left_im, right_im, net_dict, global_step):
        disp_l = net_dict['disp_l'][0].to('cpu')
        disp_r = net_dict['disp_r'][0].to('cpu')
        im_rec_l = net_dict['im_rec_l'][0].to('cpu')
        im_rec_r = net_dict['im_rec_r'][0].to('cpu')

        left_image = vutils.make_grid(left_im, normalize=True)
        right_image = vutils.make_grid(right_im, normalize=True)         
        recon_left_image = vutils.make_grid(im_rec_l, normalize=True)
        recon_right_image = vutils.make_grid(im_rec_r, normalize=True)
        dmap_left = vutils.make_grid(disp_l, normalize=True)
        dmap_right = vutils.make_grid(disp_r, normalize=True)
        self.writer.add_image('left_im', left_image, global_step)
        self.writer.add_image('right_im', right_image, global_step)
        self.writer.add_image('recon_left_im', recon_left_image, global_step)
        self.writer.add_image('recon_right_im', recon_right_image, global_step)
        self.writer.add_image('left_disp', dmap_left, global_step)
        self.writer.add_image('right_disp', dmap_right, global_step)
        print('Saved to tensorboard')

    def network_info(self):
        print('Network loaded')
        total_params = sum(p.numel() for p in self.Net.parameters())
        print('Total number of parameters: {}'.format(total_params))
        
    def sample_display(self):
        data = next(iter(self.data_loader))
        im_right = data['right'][0]
        im_left = data['left'][0]
        im_right = vutils.make_grid(im_right, normalize=True)
        im_left = vutils.make_grid(im_left, normalize=True)
        im_right = np.transpose(im_right.numpy(), (1, 2, 0))
        im_left = np.transpose(im_left.numpy(), (1, 2, 0))
        fig = plt.figure() 
        fig.add_subplot(1, 2, 1)
        plt.imshow(im_left)
        plt.title('Left image')
        fig.add_subplot(1, 2, 2)
        plt.imshow(im_right)
        plt.title('Right image')
        plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpus', type=list, default = [0])
    parser.add_argument('--model_file', type=str, default='network3')
    parser.add_argument('--losses_file', type=str, default='losses3')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--root', type=str, default='/hdd/local/sdb/umar/kitti_dataset2/')
    parser.add_argument('--image_size', type=list, default=[256, 512])

    args = parser.parse_args()

    epochs = args.epochs 
    batch_size = args.batch_size 
    gpus = args.gpus 
    model_file = args.model_file 
    losses_file = args.losses_file 
    lr = args.lr 
    beta1 = args.beta1 
    beta2 = args.beta2 
    root = args.root 
    image_size = args.image_size 

    net_module = importlib.import_module(model_file) 
    losses_module = importlib.import_module(losses_file) 

    Trainer(epochs=epochs, 
            batch_size=batch_size,
            gpus=gpus, 
            net_module=net_module, 
            losses_module=losses_module, 
            lr=lr, 
            beta1=beta1,
            beta2=beta2,
            root=root,
            image_size=image_size)


