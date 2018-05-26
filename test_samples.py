import os
import os.path as op
import numpy as np
import selector

model_path='/home/yfji/Pretrained/pytorch/vgg19.pth.2'
dataset_root='/mnt/sda6/MOT2017/train'
dirs=os.listdir(dataset_root)
dataset_dirs=[op.join(dataset_root,_dir,'img1') for _dir in dirs]
label_files=[op.join(dataset_root,_dir,'gt/gt.txt') for _dir in dirs]

data_loader=selector.ImageDataLoader(dataset_dirs)
pair_selector=selector.PairSelector(data_loader, label_files)
trip_selector=selector.TripletSelector(data_loader, label_files)

savedir='./visualize_test'
image_index=0

for i in range(50):
    print('loading %d sample pair'%i)
    pos, neg, label=pair_selector.get_data_in_batch()
