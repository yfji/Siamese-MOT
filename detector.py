import sys
from darknet import Darknet
import util
import os.path as op
import pickle
import torch
import numpy as np
import cv2
import torch
from torch.autograd import Variable

class Detector(object):
    def __init__(self, model_def_file='', weights_file=''):
        self.model_def_file=model_def_file
        self.weights_file=weights_file
        self.model=Darknet(self.model_def_file)
        self.model.load_weights(self.weights_file)
        
        self.CUDA=True
        if self.CUDA:
            self.model.cuda()
            
        print('load network finish')
        
        self.confidence=0.5
        self.nms_thresh=0.4
        self.num_classes=80
        
        self.yolo_dir='../pytorch-yolo-v3'
        self.classes=util.load_classes(op.join(self.yolo_dir, 'data/coco.names'))
        self.colors=pickle.load(open(op.join(self.yolo_dir, 'pallete'), 'rb'))
        
        #default 320x320
        self.net_input_height = int(self.model.net_info["height"])
        self.net_input_width= int(self.model.net_info["width"])
        
    def detect(self, image):
        img=self.prep_image(image)
        
        with torch.no_grad():
            prediction = self.model(img, self.CUDA)
        output=self.filter_results(prediction).cpu().numpy()  #list of [score, x1,y1,x2,y2]
        
        output[:,1:5]/=self.scale
        for i in range(output.shape[0]):
            output[i, [1,3]] = np.minimum(np.maximum(output[i, [1,3]], 0), image.shape[1])
            output[i, [2,4]] = np.minimum(np.maximum(output[i, [2,4]], 0), image.shape[0])
        
        output=output[output[:,-1]==0]
        return output    #tensor of (N,8)
        
    def prep_image(self, image):
        #RGB image
        self.scale=min(self.net_input_height*1.0/image.shape[0],self.net_input_width*1.0/image.shape[1])
        img_pad=128*np.ones((self.net_input_height, self.net_input_width, 3), dtype=np.float32)
        img=cv2.resize(image, (0,0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        img_pad[:img.shape[0],:img.shape[1],:]=img
        img_pad=img_pad[:,:,::-1].transpose(2,0,1).astype(np.float32)
        img_pad/=255.0
        img_pad=img_pad[np.newaxis,:,:,:]
        return Variable(torch.from_numpy(img_pad).float().cuda())
        
    def filter_results(self, prediction, nms=True):
        conf_mask = (prediction[:,:,4] > self.confidence).float().unsqueeze(2)
        prediction = prediction*conf_mask
    
        try:
            torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
        except:
            return 0
        
        box_a = prediction.new(prediction.shape)
        box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
        box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
        box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
        box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
        prediction[:,:,:4] = box_a[:,:,:4]
        
        batch_size = prediction.size(0)
        
        output = prediction.new(1, prediction.size(2) + 1)
        write = False
    
        for ind in range(batch_size):
            image_pred = prediction[ind]
            
            max_conf, max_conf_score = torch.max(image_pred[:,5:5+ self.num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
            seq = (image_pred[:,:5], max_conf, max_conf_score)
            image_pred = torch.cat(seq, 1)
            
            non_zero_ind =  (torch.nonzero(image_pred[:,4]))
    
            try:
                image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
            except:
                continue
            img_classes = util.unique(image_pred_[:,-1])
            
            for cls in img_classes:
                #get the detections with one particular class
                cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
                
    
                image_pred_class = image_pred_[class_mask_ind].view(-1,7)
    
                conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.size(0)
                
                if nms:
                    for i in range(idx):
                        try:
                            ious = util.bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                        except ValueError:
                            break
            
                        except IndexError:
                            break
                        
                        iou_mask = (ious < self.nms_thresh).float().unsqueeze(1)
                        image_pred_class[i+1:] *= iou_mask       
                        
                        non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                        image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                seq = batch_ind, image_pred_class
                if not write:
                    output = torch.cat(seq,1)
                    write = True
                else:
                    out = torch.cat(seq,1)
                    output = torch.cat((output,out))
        
        return output