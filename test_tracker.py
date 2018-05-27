#!/usr/bin/python3

"""
Siamese fc network for tracking

@author: yfji
"""
import torch
from torch.autograd import Variable
import torch.optim as optim
import model
import selector
import loss
import os
import os.path as op
import numpy as np
import numpy.random as nr
import cv2

class SiameseTracker:
    def __init__(self):
        self.model_path='/home/yfji/Pretrained/pytorch/vgg19.pth.2'
        self.dataset_root='/mnt/sda6/MOT2017/train'
        dirs=os.listdir(self.dataset_root)
        self.dataset_dirs=[op.join(self.dataset_root,_dir,'img1') for _dir in dirs]
        self.label_files=[op.join(self.dataset_root,_dir,'gt/gt.txt') for _dir in dirs]
        
        self.siamese=model.SiameseNet(pretrain=False, init=False)
        self.siamese.load_weights(model_path='models_siamese/model_iter_8000.pkl')
#        self.siamese=model.SimpleSiameseNet()
        self.siamese.cuda()
        self.siamese_loss=loss.OnlineContrastiveLoss(margin=20)
        self.siamese_loss.cuda()
        self.data_loader=selector.ImageDataLoader(self.dataset_dirs)
        self.selector=selector.BaseSelector(self.data_loader, self.label_files)
    
    def test(self):
        pass
    
    def gen_samples(self):
        base_dir='./test_samples'
        all_targets=self.selector.targets
        num_targets=self.selector.num_targets
        num_datasets=self.selector.num_datasets
        
        dataset_index=nr.randint(0,num_datasets)
        targets=all_targets[dataset_index]
        
        p_index=nr.randint(0,num_targets[dataset_index])
        pos_targets=targets[p_index]
        for i, t in enumerate(pos_targets):
            frame_id=t['frame_id']
            bbox=t['bbox']
            image=self.data_loader.load_image(dataset_index, frame_id)
            bbox=np.maximum(0,bbox).astype(np.int32)
            coords=[bbox[0],bbox[1],min(image.shape[1],bbox[0]+bbox[2]),min(image.shape[0], bbox[1]+bbox[3])]
        
            im=image[coords[1]:coords[3],coords[0]:coords[2],:]
            cv2.imwrite(op.join(base_dir, 'pos_%d.jpg'%i), im)
        for i in range(len(pos_targets)):
            n_index=p_index
            while(n_index==p_index):
                n_index=nr.randint(0,num_targets[dataset_index])
            n_targets=targets[n_index]
            n_t=n_targets[nr.randint(0,len(n_targets))]
            
            frame_id=n_t['frame_id']
            bbox=n_t['bbox']
            
            image=self.data_loader.load_image(dataset_index, frame_id)
            bbox=np.maximum(0,bbox).astype(np.int32)
            coords=[bbox[0],bbox[1],min(image.shape[1],bbox[0]+bbox[2]),min(image.shape[0], bbox[1]+bbox[3])]
        
            im=image[coords[1]:coords[3],coords[0]:coords[2],:]
            cv2.imwrite(op.join(base_dir, 'neg_%d.jpg'%i), im)
    
    def run(self, mode='train'):
        self.gen_samples()

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DIVICES']='0'
    tracker=SiameseTracker()
    tracker.run()