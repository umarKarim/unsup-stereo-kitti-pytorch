import torch
import os
import torchvision.transforms as transforms 
import PIL.Image as Image 
import random
import matplotlib.pyplot as plt
import numpy as np


class StereoLoader():
    def __init__(self, root='/hdd/local/sdb/umar/kitti_dataset2/', train=True, req_transforms=None,
                image_size=[256, 512], im_names='stereo_im_names.npy'):
        self.root = root
        self.train = train 
        self.transforms = transforms.Compose(req_transforms) 
        self.image_size = image_size 
        self.im_names = im_names

        if not os.path.exists(self.im_names):
            # accessing sequences 
            self.sequences = os.listdir(root)
            self.sequences = [x for x in self.sequences if os.path.isdir(root + x)]
            self.sequences.sort()

            # acessing left and right videos
            self.left_images = []
            self.right_images = []
            for s in self.sequences:
                left_images_curr = os.listdir(self.root + s + '/image_02/data/')
                left_images_curr = [x for x in left_images_curr if x.endswith('.png') and x.startswith('0')]
                for i, _ in enumerate(left_images_curr):
                    left_images_curr[i] = self.root + s + '/image_02/data/' + left_images_curr[i] 
                self.left_images += left_images_curr
                self.left_images.sort()

                right_images_curr = os.listdir(self.root + s + '/image_03/data/')
                right_images_curr = [x for x in right_images_curr if x.endswith('.png') and x.startswith('0')]
                for i, _ in enumerate(right_images_curr):
                    right_images_curr[i] = self.root + s + '/image_03/data/' + right_images_curr[i]
                self.right_images += right_images_curr
                self.right_images.sort()

            data = {'left_names': self.left_images, 'right_names': self.right_images}
            np.save(self.im_names, data) 

        else:
            data = np.load(self.im_names, allow_pickle=True).item()
            self.left_images = data['left_names']
            self.right_images = data['right_names']
        
        assert len(self.left_images) == len(self.right_images), 'Total images are not equal'
        assert len(self.left_images) != 0, 'Images not read'

    def display_sample(self):
        i = random.randint(0, len(self.left_images))
        left_im = Image.open(self.left_images[i])
        right_im = Image.open(self.right_images[i])
        print(self.left_images[i])
        print(self.right_images[i])
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(left_im)
        plt.subplot(1, 2, 2)
        plt.imshow(right_im)
        plt.show()

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, index):
        left_im = self.transforms(Image.open(self.left_images[index]))
        right_im = self.transforms(Image.open(self.right_images[index]))
        return {'left': left_im, 'right': right_im}


if __name__ == '__main__':
    transform = [transforms.Resize((256, 512)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataset = StereoLoader(req_transforms=transform)
    dataset.display_sample()
    len(dataset)
    images = dataset[1]
    print(images['left'].size())
    print(images['right'].size())


