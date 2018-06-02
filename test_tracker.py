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

class Tracker:
    def __init__(self):
        self.model_path='/home/yfji/Pretrained/pytorch/vgg19.pth.2'
        self.dataset_root='/mnt/sda6/MOT2017/train'
        self.sample_dir='./test_samples'

        dirs=os.listdir(self.dataset_root)
        self.dataset_dirs=[op.join(self.dataset_root,_dir,'img1') for _dir in dirs]
        self.label_files=[op.join(self.dataset_root,_dir,'gt/gt.txt') for _dir in dirs]
        
        self.net=model.SiameseNet(pretrain=False, init=False)
        self.net.load_weights(model_path='models_siamese/model_iter_80000.pkl')
        self.net.cuda()
    
    def test(self):
        branch1_names=[]
        branch2_names=[]
        self.batch_size=16
        
        sample_names=os.listdir(self.sample_dir)
        for name in sample_names:
            if 'pos' in name:
                branch1_names.append(op.join(self.sample_dir, name))
            elif 'neg' in name:
                branch2_names.append(op.join(self.sample_dir, name))
        branch1_index=nr.choice(np.arange(len(branch1_names)), size=self.batch_size, replace=False)
        branch2_index=nr.choice(np.arange(len(branch2_names)), size=self.batch_size, replace=False)
        
        branch1_select=[]
        branch2_select=[]
        for i in range(self.batch_size):
            branch1_select.append(branch1_names[branch1_index[i]])
            branch2_select.append(branch2_names[branch2_index[i]])
        
        x1,x2,names1,names2=self.gen_batch_data(branch1_select, branch2_select)
        
        x1=Variable(torch.FloatTensor(x1).cuda())
        x2=Variable(torch.FloatTensor(x2).cuda())
        
        dist_l2, dist_l1=self.net.pairwise_distance(x1, x2)
        
        dist_l2=dist_l2.data.cpu().numpy()
        dist_l1=dist_l1.data.cpu().numpy()
        
        dist=dist_l1
        
        for i in range(self.batch_size):
            print('%s:%s: %f'%(names1[i],names2[i],dist[i]))
        
        pos_dist=[]
        neg_dist=[]
        
        for i in range(self.batch_size):
            prop1=names1[:names1.rfind('_')]
            prop2=names2[:names2.rfind('_')]
            if prop1=='pos' and prop2=='pos':
                pos_dist.append(dist[i])
            else:
                neg_dist.append(dist[i])
        print('pos mean dist: %f, neg mean dist: %f'%(np.mean(pos_dist), np.mean(neg_dist)))
        print('pos max dist: %f, neg min dist: %f'%(np.max(pos_dist), np.min(neg_dist)))
                
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
            image-=image.min()
            image/=image.max()
            x1[i]=image.transpose(2,0,1)
            name=all_names[random_order[i]]
            batch_names1.append(name[name.rfind('/')+1:])
            
        for i in range(self.batch_size, self.batch_size*2):
            image=cv2.imread(all_names[random_order[i]])
            image=cv2.resize(image,(sz,sz),interpolation=cv2.INTER_CUBIC).astype(np.float32)
            image-=image.min()
            image/=image.max()
            x2[i-self.batch_size]=image.transpose(2,0,1)
            name=all_names[random_order[i]]
            batch_names2.append(name[name.rfind('/')+1:])
        
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
    tracker=Tracker()
    tracker.run()