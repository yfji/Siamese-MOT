'''
Selectors for pair and triplet
'''

from itertools import combinations

import numpy as np
import torch
import cv2
import numpy as np
import numpy.random as nr
import os
import os.path as op

#=============DataLoader===============#   
class DataLoader():
    def __init__(self):
        pass
    
    def load_image(self,dataset_index, img_index):
        pass

class ImageDataLoader(DataLoader):
    def __init__(self, image_dirs):
        super(ImageDataLoader,self).__init__()
        self.image_dirs=image_dirs
        self.num_datasets=len(self.image_dirs)
        
    def load_image(self, dataset_index, img_index):
        img_names=os.listdir(self.image_dirs[dataset_index])
        img_name=img_names[img_index]
        assert(os.path.exists(op.join(self.image_dirs[dataset_index], img_name)))
        return cv2.imread(op.join(self.image_dirs[dataset_index], img_name))
        
        
class VideoDataLoader(DataLoader):
    def __init__(self, video_files):
        super(VideoDataLoader, self).__init__()
        self.video_files=video_files
        self.caps=[cv2.VideoCapture(video_file) for video_file in self.video_files]
        self.num_datasets=len(self.caps)
    
    def load_image(self, cap_index, frame_index):
        self.caps[cap_index].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok,frame=self.cap.read()
        if not ok:
            return np.zeros()
        else:
            return frame

#=============Selector===============#      
class BaseSelector():
    def __init__(self, data_loader, label_files):
        self.data_loader=data_loader
        self.labels=label_files
        self.num_datasets=len(self.labels)
        assert(self.num_datasets==self.data_loader.num_datasets)
        
        self.batch_size=8
        self.net_input_size=128
        self.cat_conf_thresh=0.6
        self.same_cat_prob=0.5

        self.parse_labels()
        
    def parse_label(self, dataset_index):
        assert(op.exists(self.labels[dataset_index]))
        label=open(self.labels[dataset_index],'r')
        lines=label.readlines()
        targets=[]
        target=[]
        cur_target_id=-1
        for line in lines:
            if len(line)<1:
                continue
            items=line.rstrip().split(',')
            items=list(map(float,items))
            target_id=int(items[1])
            frame_id=int(items[0])
            cat_id=int(items[6])
            score=items[8]
            if target_id!=cur_target_id:
                cur_target_id=target_id
                if cur_target_id!=-1 and len(target)>=3:
                    targets.append(target)
                target=[]
            if cat_id==1 and score>self.cat_conf_thresh:
                t_entry={}
                t_entry['target_id']=target_id-1
                t_entry['frame_id']=frame_id-1
                t_entry['bbox']=items[2:6]  #x1,y1,w,h
                target.append(t_entry)
        if len(target)>=3:
            targets.append(target)
        num_targets=len(targets)
#        data_indices=np.arange(self.num_targets)
        label.close()
        return targets, num_targets
    
    def parse_labels(self):
        self.targets=[]
        self.num_targets=[]
        
        for i in range(self.num_datasets):
            targets, num_target=self.parse_label(i)
            self.targets.append(targets)
            self.num_targets.append(num_target)
        print(self.num_targets)
    
    def get_data(self):
        '''
        return a pair of either positive samples or negative samples
        '''
        pass
    
class PairSelector(BaseSelector):
    def __init__(self, data_loader, label_files):
        super(PairSelector, self).__init__(data_loader, label_files)
        self.visualize=True
        self.savedir='./visualize_siamese'
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        self.image_index=0
    
    def get_data_in_batch(self):
        same_prob=nr.rand()
        dataset_index=nr.randint(0,self.num_datasets)
        targets=[]
        target_indices=[]
        sample_indices=[]
        y=0.0
        if same_prob>self.same_cat_prob:
#        if same_prob>1:
            idx=nr.randint(0,self.num_targets[dataset_index])
            target_indices=[idx,idx]
            targets=[self.targets[dataset_index][target_indices[0]],self.targets[dataset_index][target_indices[1]]]
            sample_indices=nr.choice(np.arange(len(targets[0])),size=2,replace=False)
            y=1.0
        else:
            target_indices=nr.choice(np.arange(self.num_targets[dataset_index]), size=2, replace=False)
            targets=[self.targets[dataset_index][target_indices[0]],self.targets[dataset_index][target_indices[1]]]
#            print(targets)
            sample_indices=np.asarray([nr.randint(0,len(targets[0])), nr.randint(0,len(targets[1]))])
       
        bbox1=targets[0][sample_indices[0]]['bbox']
        bbox2=targets[1][sample_indices[1]]['bbox']
        bbox1=np.maximum(0,bbox1).astype(np.int32)
        bbox2=np.maximum(0,bbox2).astype(np.int32)
        
        image1=self.data_loader.load_image(dataset_index,targets[0][sample_indices[0]]['frame_id'])
        image2=self.data_loader.load_image(dataset_index,targets[1][sample_indices[1]]['frame_id'])
        im_t1=image1[bbox1[1]:min(image1.shape[0],bbox1[1]+bbox1[3]),bbox1[0]:min(image1.shape[1], bbox1[0]+bbox1[2]),:]
        im_t2=image1[bbox2[1]:min(image2.shape[0],bbox2[1]+bbox2[3]),bbox2[0]:min(image2.shape[1], bbox2[0]+bbox2[2]),:]
        
        im_t1=cv2.resize(im_t1,(self.net_input_size,self.net_input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        im_t2=cv2.resize(im_t2,(self.net_input_size,self.net_input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        
        if self.visualize and self.image_index<=50:
            mid_pad=10
            image_merge=128*np.ones((self.net_input_size, 2*self.net_input_size+mid_pad,3), dtype=np.uint8)
            image_merge[:,:self.net_input_size,:]=im_t1
            image_merge[:,self.net_input_size+mid_pad:,:]=im_t2
            if y==1:
                name=op.join(self.savedir, 'pos_%d.jpg'%self.image_index)
            else:
                name=op.join(self.savedir, 'neg_%d.jpg'%self.image_index)
            cv2.imwrite(name, image_merge)
            self.image_index+=1
                
        im_t1=(im_t1-128.0)/255.0
        im_t2=(im_t2-128.0)/255.0
        
        return im_t1,im_t2,y
    
    def get_data(self):
        data=np.zeros((self.batch_size, 2, 3,self.net_input_size, self.net_input_size), dtype=np.float32)
        label=np.zeros(self.batch_size, dtype=np.float32)
        for i in range(self.batch_size):
            im_t1,im_t2,y=self.get_data_in_batch()
            data[i,0,:,:,:]=im_t1.transpose(2,0,1)
            data[i,1,:,:,:]=im_t2.transpose(2,0,1)
            label[i]=y
        return data, label
        
class TripletSelector(BaseSelector):
    def __init__(self, data_loader, label_files):
        super(TripletSelector, self).__init__(data_loader, label_files)
        self.visualize=True
        self.savedir='./visualize_triplet'
        self.image_index=0
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
    
    def get_data_in_batch(self):
        dataset_index=nr.randint(0,self.num_datasets)

        target_indices=nr.choice(np.arange(self.num_targets[dataset_index]), size=2, replace=False)
        targets=[self.targets[target_indices[0]],self.targets[target_indices[1]]]
        sample_indices=np.hstack((nr.choice(np.arange(len(targets[0])),size=2,replace=False), nr.randint(0,len(targets[1]))))
        anchor_bbox=targets[0][sample_indices[0]]['bbox']
        pos_bbox=targets[0][sample_indices[1]]['bbox']
        neg_box=targets[1][sample_indices[2]]['bbox']
        
        anchor_bbox=np.maximum(0,anchor_bbox).astype(np.int32)
        pos_bbox=np.maximum(0,pos_bbox).astype(np.int32)
        neg_bbox=np.maximum(0,neg_box).astype(np.int32)
        
        anchor_image=self.data_loader.load_image(self.cap_index,targets[0][sample_indices[0]]['frame_id'])
        pos_image=self.data_loader.load_image(dataset_index,targets[0][sample_indices[1]]['frame_id'])
        neg_image=self.data_loader.load_image(dataset_index,targets[1][sample_indices[2]]['frame_id'])
        
        im_anchor=anchor_image[anchor_bbox[1]:min(anchor_image.shape[0],anchor_bbox[1]+anchor_bbox[3]),anchor_bbox[0]:min(anchor_image.shape[1],anchor_bbox[0]+anchor_bbox[2]),:]
        im_pos=pos_image[pos_bbox[1]:min(pos_image.shape[0],pos_bbox[1]+pos_bbox[3]),pos_bbox[0]:min(pos_image.shape[1],pos_bbox[0]+pos_bbox[2]),:]
        im_neg=pos_image[neg_bbox[1]:min(neg_image.shape[0],neg_bbox[1]+neg_bbox[3]),neg_bbox[0]:min(neg_image.shape[1],neg_bbox[0]+neg_bbox[2]),:]
        
        im_anchor=cv2.resize(im_anchor,(self.net_input_size,self.net_input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        im_pos=cv2.resize(im_pos,(self.net_input_size,self.net_input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        im_neg=cv2.resize(im_neg,(self.net_input_size,self.net_input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        if self.visualize and self.image_index<=50:
            mid_pad=10
            image_merge=128*np.ones((self.net_input_size, 3*self.net_input_size+2*mid_pad,3), dtype=np.uint8)
            image_merge[:,:self.net_input_size,:]=im_anchor
            image_merge[:,self.net_input_size+mid_pad:,:]=im_pos
            image_merge[:,2*(self.net_input_size+mid_pad):,:]=im_neg
            
            name=op.join(self.savedir, 'triplet_%d.jpg'%self.image_index)
            cv2.imwrite(name, image_merge)
            self.image_index+=1

        im_anchor=(im_anchor-128.0)/255.0
        im_pos=(im_pos-128.0)/255.0
        im_neg=(im_neg-128.0)/255.0
        
        return im_anchor,im_pos,im_neg
    
    def get_data(self):
        data=np.zeros((self.batch_size, 3, 3,self.net_input_size, self.net_input_size),dtype=np.float32)
        for i in range(self.batch_size):
            im_anchor,im_pos,im_neg=self.get_data_in_batch()
            data[i,0,:,:,:]=im_anchor.transpose(2,0,1)
            data[i,1,:,:,:]=im_pos.transpose(2,0,1)
            data[i,2,:,:,:]=im_neg.transpose(2,0,1)
        return data

        