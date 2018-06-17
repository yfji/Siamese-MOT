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


class SiameseTracker:
    def __init__(self):
#        self.model_path='/home/yfji/Pretrained/pytorch/vgg19.pth.2'
        self.dataset_root='/mnt/sda6/MOT2017/train'
        dirs=os.listdir(self.dataset_root)
        self.dataset_dirs=[op.join(self.dataset_root,_dir,'img1') for _dir in dirs]
        self.label_files=[op.join(self.dataset_root,_dir,'gt/gt.txt') for _dir in dirs]
        
        self.siamese=model.SiameseNet(init=True)
#        self.siamese=model.SimpleSiameseNet()
        self.siamese.cuda()
        self.siamese_loss=loss.OnlineContrastiveLoss(margin=40)
        self.siamese_loss.cuda()
        data_loader=selector.ImageDataLoader(self.dataset_dirs)
        self.pair_selector=selector.PairSelector(data_loader, self.label_files)
    
    def train(self):
        max_iter=100000
        lr=0.000008
        decay_ratio=0.1
        display=20
        snapshot=20000
        step_index=0
        stepvalues=[60000,100000]
        g_steps=stepvalues[0]
        
        param_groups=[]
        for key, value in self.siamese.named_parameters():
            if value.requires_grad:
                param_groups.append({'params': value, 'lr': lr})
            
        optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9)

        step_index=0
        step=0
        for i in range(max_iter):
            pair_samples, y_np=self.pair_selector.get_data()
            
            pos_samples=Variable(torch.FloatTensor(pair_samples[:,0,:,:,:]).cuda())   #[N,C,H,W]
            neg_samples=Variable(torch.FloatTensor(pair_samples[:,1,:,:,:]).cuda())
            y=Variable(torch.FloatTensor(y_np).cuda())
#            y=torch.FloatTensor(y).contiguous().cuda(async=True)
            
            pos_feat, neg_feat=self.siamese(pos_samples, neg_samples)
            
            loss, dist=self.siamese_loss(pos_feat,neg_feat,y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            rate=lr*np.power(decay_ratio,step/g_steps)
#            for param_group in optimizer.param_groups:
#                param_group['lr']=rate        
            if i%display==0:
                print('[Info][%d/%d] loss: %f, learn rate: %e'%(i,max_iter,loss, lr))
                dist=dist.data.cpu().numpy()
                pos_labels=(y_np==1)
                neg_labels=(y_np==0)
                
                pos_dist=np.mean(dist[pos_labels],axis=0) if len(np.nonzero(pos_labels)[0])>0 else 0 
                neg_dist=np.mean(dist[neg_labels],axis=0) if len(np.nonzero(neg_labels)[0])>0 else 0 
                print('pos pair dist: %f\nneg pair dist: %f'%(pos_dist,neg_dist))
            if i==stepvalues[step_index]:
                for param_group in optimizer.param_groups:
                    param_group['lr']=rate   
                print('learn rate decay: %e'%rate)
                step=0
                lr=rate
                g_steps=stepvalues[step_index+1]-stepvalues[step_index]
                step_index+=1
            if i>0 and i%snapshot==0:
                torch.save(self.siamese.state_dict(), 'models_siamese/model_iter_%d.pkl'%i)
                print('Snapshot to models_siamese/model_iter_%d.pkl'%i)
            step+=1
        torch.save(self.siamese.state_dict(), 'models_siamese/model_iter_%d.pkl'%max_iter)
    
    def test(self):
        pass
    
    def run(self, mode='train'):
        self.train()

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DIVICES']='0'
    tracker=SiameseTracker()
    tracker.run()