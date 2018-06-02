#!/usr/bin/python3

"""
Siamese fc network for tracking

@author: yfji
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from math import sqrt
from vgg import Vgg


class EmbeddingNet(nn.Module):
    def __init__(self, pretrain=False, init=False):
        super(EmbeddingNet,self).__init__()
        self.vgg=Vgg()
        self.vgg.make_vgg()
        self.net_input_size=128
        self.stride=self.vgg.backbone_stride
        
        self.maxpool=nn.MaxPool2d(2, stride=2)
        self.stride*=2
        blocks=[]
        conv_out=(self.net_input_size/self.stride)**2*256
        blocks.append(nn.Linear(conv_out, 512))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Linear(512, 256))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Linear(256, 2))
        
        self.task=nn.Sequential(*blocks)
        
        if not pretrain and init:
            self.init_weights(self)
        elif pretrain:
            self.vgg.load_backbone(model_path='/home/yfji/Pretrained/pytorch/vgg19.pth.2')
            self.vgg.apply_fix()
            self.init_weights(self.task)
    
    def load_weights(self, model_path=''):
        model_dict = self.state_dict()
        print('loading model from {}'.format(model_path))
        try:
            pretrained_dict = torch.load(model_path)
            tmp = OrderedDict()
            for k,v in pretrained_dict.items():
                if k in model_dict:
                    tmp[k] = v
            model_dict.update(tmp)
            self.load_state_dict(model_dict)
        except:
            print ('loading model failed, {} may not exist'.format(model_path))
            
    def init_weights(self, module):
        for m in module.modules():
            if isinstance(m,nn.Conv2d):
                m.weight.data.normal_(0,0.1)
#                torch.nn.init.xavier_uniform_(m.weight.data, gain=sqrt(2.0))
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(0,0.1)
#                torch.nn.init.xavier_uniform_(m.weight.data, gain=sqrt(2.0))
                m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def pairwise_distance(self, x1, x2):
        out1=self.vgg(x1)
        out1=out1.view(out1.size()[0],-1)
        out1=self.task(out1)
        
        out2=self.vgg(x2)
        out2=out2.view(out2.size()[0],-1)
        out2=self.task(out2)
        
        dist_l2=torch.pow(out1-out2, 2).sum(1)
        dist_l1=torch.sqrt(dist_l2)
        return dist_l2, dist_l1    
    
    def forward(self, x):
        out=self.vgg(x)
        out=self.maxpool(out)
        out=out.view(out.size()[0],-1)
        out=self.task(out)
        
        return out
    
class SimpleEmbeddingNet(nn.Module):
    def __init__(self, init=False):
        super(SimpleEmbeddingNet,self).__init__()
        self.net_input_size=128
        self.out_channels=64
        self.out_dims=32
        convs=[]
        convs+=self.make_layer(3,56)    #downsample 2
        convs+=self.make_layer(56,56)   #downsample 4
        convs+=self.make_layer(56,64)   #downsample 8
        convs+=self.make_layer(64,self.out_channels)   #downsample 16-->8
            
        self.stride=2**4
        
        self.backbone=nn.Sequential(*convs)
        
        fcs=[]
        conv_out=(self.net_input_size/self.stride)**2*self.out_channels
        fcs.append(nn.Linear(conv_out, 256))
        fcs.append(nn.ReLU(inplace=True))
        fcs.append(nn.Linear(256, 256))
        fcs.append(nn.ReLU(inplace=True))
        fcs.append(nn.Linear(256,self.out_dims))
        
        self.task=nn.Sequential(*fcs)
        
        if init:
            self.init_weights(self.backbone)
            self.init_weights(self.task)
    
    def make_layer(self, in_channels, out_channels, thickness=2, relu=True, downsample=True):
        convs=[]
        convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
        if relu:
            convs.append(nn.ReLU(inplace=True))
        for i in range(1,thickness):
            convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
            if relu:
                convs.append(nn.ReLU(inplace=True))
        if downsample:
            convs.append(nn.MaxPool2d(2,stride=2))
        return convs
        
    def init_weights(self, module):
        for m in module.modules():
            if isinstance(m,nn.Conv2d):
                print('conv2d')
                m.weight.data.normal_(0,0.1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                print('linear')
                m.weight.data.normal_(0,0.1)
                m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                print('batchnorm')
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def load_weights(self, model_path=None):
        model_dict = self.state_dict()
        print('loading model from {}'.format(model_path))
        try:
            pretrained_dict = torch.load(model_path)
            tmp = OrderedDict()
            for k,v in pretrained_dict.items():
                if k in model_dict:
                    tmp[k] = v
            model_dict.update(tmp)
            self.load_state_dict(model_dict)
        except:
            print ('loading model failed, {} may not exist'.format(model_path))
    
    def forward(self, x):
        out=self.backbone(x)
        out=out.view(out.size()[0],-1)
        out=self.task(out)
        return out
        
    def pairwise_distance(self, x1, x2):
        out1=self.backbone(x1)
        out1=out1.view(out1.size()[0],-1)
        out1=self.task(out1)
        
        out2=self.backbone(x2)
        out2=out2.view(out2.size()[0],-1)
        out2=self.task(out2)
        
        dist_l2=torch.pow(out1-out2, 2).sum(1)
        dist_l1=torch.sqrt(dist_l2)
        return dist_l2, dist_l1
        
        
class SiameseNet(nn.Module):
    def __init__(self, pretrain=False, init=False):
        super(SiameseNet, self).__init__()
        self.embedding=SimpleEmbeddingNet(init=init)
        self.net_input_size=self.embedding.net_input_size
    
    def load_weights(self, model_path=''):
        model_dict = self.state_dict()
        print('loading model from {}'.format(model_path))
        try:
            pretrained_dict = torch.load(model_path)
            tmp = OrderedDict()
            for k,v in pretrained_dict.items():
                if k in model_dict:
                    tmp[k] = v
            model_dict.update(tmp)
            self.load_state_dict(model_dict)
        except:
            print ('loading model failed, {} may not exist'.format(model_path))
    
    def pairwise_distance(self, x1, x2):
        return self.embedding.pairwise_distance(x1, x2)
    
    def forward(self, x1, x2):
        out1=self.embedding(x1)
        out2=self.embedding(x2)
        
        return out1, out2
    
class TripletNet(nn.Module):
    def __init__(self, pretrain=False, init=False):
        super(TripletNet,self).__init__()
        self.embedding=SimpleEmbeddingNet(init=init)
        self.net_input_size=self.embedding.net_input_size
    
    def load_weights(self, model_path=''):
        model_dict = self.state_dict()
        print('loading model from {}'.format(model_path))
        try:
            pretrained_dict = torch.load(model_path)
            tmp = OrderedDict()
            for k,v in pretrained_dict.items():
                if k in model_dict:
                    tmp[k] = v
            model_dict.update(tmp)
            self.load_state_dict(model_dict)
        except:
            print ('loading model failed, {} may not exist'.format(model_path))
            
                
    def pairwise_distance(self, x1, x2):
        return self.embedding.pairwise_distance(x1, x2)

    def forward(self, x1,x2,x3):
        out1=self.embedding(x1)
        out2=self.embedding(x2)
        out3=self.embedding(x3)
        
        return out1, out2, out3