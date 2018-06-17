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


class TripletTracker:
    def __init__(self):
#        self.model_path='/home/yfji/Pretrained/pytorch/vgg19.pth.2'
        self.dataset_root='/mnt/sda6/MOT2017/train'
        dirs=os.listdir(self.dataset_root)
        self.dataset_dirs=[op.join(self.dataset_root,_dir,'img1') for _dir in dirs]
        self.label_files=[op.join(self.dataset_root,_dir,'gt/gt.txt') for _dir in dirs]
        
        self.triplet=model.TripletNet(init=True)
#        self.siamese=model.SimpleSiameseNet()
        self.triplet.cuda()
        self.trip_loss=loss.OnlineTripletLoss(margin=40)
        self.trip_loss.cuda()
        data_loader=selector.ImageDataLoader(self.dataset_dirs)
        self.trip_selector=selector.TripletSelector(data_loader, self.label_files)
    
    def train(self):
        max_iter=100000
        lr=0.000002
        decay_ratio=0.1
        display=20
        snapshot=20000
        step_index=0
        stepvalues=[70000,100000]
        g_steps=stepvalues[0]
        
        param_groups=[]
        for key, value in self.triplet.named_parameters():
            if value.requires_grad:
                param_groups.append({'params': value, 'lr': lr})
            
        optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9)

        step_index=0
        step=0
        for i in range(max_iter):
            trip_samples=self.trip_selector.get_data()
            
            anchor_samples=Variable(torch.FloatTensor(trip_samples[:,0,:,:,:]).cuda())   #[N,C,H,W]
            pos_samples=Variable(torch.FloatTensor(trip_samples[:,1,:,:,:]).cuda())
            neg_samples=Variable(torch.FloatTensor(trip_samples[:,2,:,:,:]).cuda())
#            y=torch.FloatTensor(y).contiguous().cuda(async=True)
            
            anchor_feat, pos_feat, neg_feat=self.triplet(anchor_samples, pos_samples, neg_samples)
            
            loss, ap_dist, an_dist=self.trip_loss(anchor_feat, pos_feat, neg_feat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            rate=lr*np.power(decay_ratio,step/g_steps)
            #for param_group in optimizer.param_groups:
            #    param_group['lr']=rate        
            if i%display==0:
                print('[Info][%d/%d] loss: %f, learn rate: %e'%(i,max_iter,loss, lr))
                ap_dist=ap_dist.data.cpu().numpy()
                an_dist=an_dist.data.cpu().numpy()
                print('anchor_pos pair dist: %f\nanchor_neg pair dist: %f'%(ap_dist,an_dist))
            if i==stepvalues[step_index]:
                for param_group in optimizer.param_groups:
                    param_group['lr']=rate        
                print('learn rate decay: %e'%rate)
                step=0
                lr=rate
                g_steps=stepvalues[step_index+1]-stepvalues[step_index]
                step_index+=1
            if i>0 and i%snapshot==0:
                torch.save(self.triplet.state_dict(), 'models_triplet/model_iter_%d.pkl'%i)
                print('Snapshot to models_siamese/model_iter_%d.pkl'%i)
            step+=1
        torch.save(self.triplet.state_dict(), 'models_triplet/model_iter_%d.pkl'%max_iter)
    
    def test(self):
        pass
    
    def run(self, mode='train'):
        self.train()

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DIVICES']='0'
    tracker=TripletTracker()
    tracker.run()