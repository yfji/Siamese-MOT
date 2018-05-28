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
        self.sample_dir='./test_samples'

        dirs=os.listdir(self.dataset_root)
        self.dataset_dirs=[op.join(self.dataset_root,_dir,'img1') for _dir in dirs]
        self.label_files=[op.join(self.dataset_root,_dir,'gt/gt.txt') for _dir in dirs]
        
        self.net=model.EmbeddingNet(pretrain=False, init=False)
        self.net.load_weights(model_path='models_siamese/model_iter_8000.pkl')
#        self.siamese=model.SimpleSiameseNet()
        self.net.cuda()
        self.loss=loss.OnlineContrastiveLoss(margin=20)
        self.loss.cuda()
        self.data_loader=selector.ImageDataLoader(self.dataset_dirs)
        self.selector=selector.BaseSelector(self.data_loader, self.label_files)
    
    def test(self):
        pos_names=[]
        neg_names=[]
        self.batch_size=16
        
        sample_names=os.listdir(self.sample_dir)
        for name in sample_names:
            if 'pos' in name:
                pos_names.append(op.join(self.sample_dir, name))
            elif 'neg' in name:
                neg_names.append(op.join(self.sample_dir, name))
        pos_index=nr.choice(np.arange(len(pos_names)), size=self.batch_size, replace=False)
        neg_index=nr.choice(np.arange(len(pos_names)), size=self.batch_size, replace=False)
        
        pos_select=[]
        neg_select=[]
        for i in range(self.batch_size):
            pos_select.append(pos_names[pos_index[i]])
            neg_select.append(neg_names[neg_index[i]])
        
        x1,x2,names1,names2=self.gen_batch_data(pos_select, neg_select)
        x1=Variable(torch.FloatTensor(x1).cuda())
        x2=Variable(torch.FloatTensor(x2).cuda())
        
        feat1, feat2=self.net(x1), self.net(x2)
        
        dist=(feat1-feat2).data.cpu().numpy()
        dist_l2=np.sum(np.power(dist,2),axis=1)
        dist_l1=np.sqrt(dist_l2)
        
        for i in range(self.batch_size):
            print('%s:%s: %f'%(names1[i],names2[i],dist_l1[i]))
                
    def gen_batch_data(self, pos_names, neg_names):
        all_names=pos_names+neg_names
        random_order=nr.permutation(np.arange(len(all_names)))
        
        batch_names1=[]
        batch_names2=[]
        
        sz=self.net.net_input_size
        x1=np.zeros((self.batch_size, 3, sz,sz), dtype=np.float32)
        x2=np.zeros((self.batch_size, 3, sz,sz), dtype=np.float32)
        for i in range(self.batch_size):
            image=cv2.imread(all_names[random_order[i]])
            image=cv2.resize(image,(sz,sz),interpolation=cv2.INTER_CUBIC).astype(np.float32)
            image/=128.0;image/=255.0
            x1[i]=image.transpose(2,0,1)
            batch_names1.append(all_names[random_order[i]])
            
        for i in range(self.batch_size, self.batch_size*2):
            image=cv2.imread(all_names[random_order[i]])
            image=cv2.resize(image,(sz,sz),interpolation=cv2.INTER_CUBIC).astype(np.float32)
            image/=128.0;image/=255.0
            x2[i-self.batch_size]=image.transpose(2,0,1)
            batch_names2.append(all_names[random_order[i]])
        
        return x1,x2,batch_names1,batch_names2

        
    
    def gen_samples(self):
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
            cv2.imwrite(op.join(self.sample_dir, 'pos_%d.jpg'%i), im)
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
            cv2.imwrite(op.join(self.sample_dir, 'neg_%d.jpg'%i), im)
        
    
    
    def run(self, gen_samples=False):
        if gen_samples:
            self.gen_samples()
        self.test()
            

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DIVICES']='0'
    tracker=SiameseTracker()
    tracker.run()