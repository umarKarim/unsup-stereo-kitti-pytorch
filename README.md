# Unsupervised Stereo Disparity Estimation on KITTI Dataset
## Introduction 
This repository is for unsupervised disparity estimation on KITTI dataset. Code is clean, and good for experimental or educational purpose to understand unsupervised disparity estimation with stereo cameras. 

## Requirements 
- PyTorch 
- Torchvision 
- NumPy 
- Matplotlib 

## Usage 
Download the raw kitti dataset. 

For training, use the following code. 
```
python3 tain.py --input_dir <path_to_train_directory>
```
The intermediate qualitative results are stored in the *curr_res* directory. The losses are displayed on the console. Tensorboard can be included with ease; however, I had issues with X on my server.

For qualitative results, use the following code.
```
python3 test.py --input_dir <path_to_test_directory> --model <path_to_stored_mode>
```
The results are stored in *output_results*.

## Discussion 
There are serious generalization issues with this and similar approaches.