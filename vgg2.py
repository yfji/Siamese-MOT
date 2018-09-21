#!/usr/bin/python3

"""
Siamese fc network for tracking

@author: yfji
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from math import sqrt

class Vgg(nn.Module):
    def __init__(self):
        super(Vgg,self).__init__()
        self.backbone_stride=1

    def make_vgg(self):
        self.thickness=[2,2,4,3]
        self.layer1=self.make_layer(3,64,thickness=2)   #conv1_1,conv1_2
        self.layer2=self.make_layer(64,128,thickness=2) #conv2_1,conv2_2
        self.layer3=self.make_layer(128,256,thickness=4)    #conv3_1~conv3_4
        self.layer4=self.make_layer(256,512,thickness=4)    #conv4_1~conv4_4
        self.layer5=self.make_layer(512,256,downsample=False,thickness=2)
        
        self.show_structure()
    
    def make_layer(self, in_channels, out_channels, downsample=True, thickness=1):
        convs=[]
        convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=True))
        convs.append(nn.ReLU(inplace=True))
        for i in range(1,thickness):
            convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=True))
            convs.append(nn.ReLU(inplace=True))
        if downsample:
            convs.append(nn.MaxPool2d(2, stride=2))
            self.backbone_stride*=2
        return nn.Sequential(*convs)
     
    
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)    #8x downsample
        x=self.layer5(x)
        return x
    
    def size2str(self,size):
        if len(size)==1:
            return str(size[0])
        return '(%d,%d)'%(size[0],size[1])
    
    def load_backbone(self, model_path=None):
        model_dict=self.state_dict()
        param_keys=list(model_dict.keys())
#        print(param_keys)
        print('loading model from {}'.format(model_path))
        try:
            pretrained_dict = torch.load(model_path)
            backbone_keys=list(pretrained_dict.keys())
    #            print(backbone_keys)
    #        for k,v in pretrained_dict.items():
    #            print(k,v.size())
            offset=0
            for th in self.thickness:
                for ix in range(th):
                    param_weights=param_keys[(ix+offset)*2]
                    param_bias=param_keys[(ix+offset)*2+1]
                    backbone_weights=backbone_keys[(ix+offset)*2]
                    backbone_bias=backbone_keys[(ix+offset)*2+1]
                    
                    print('Recovering')
                    print('%s-->%s'%(param_weights,self.size2str(model_dict[param_weights].shape)))
                    print('%s-->%s'%(param_bias,self.size2str(model_dict[param_bias].shape)))
                    print('from')
                    print('%s-->%s'%(backbone_weights,self.size2str(pretrained_dict[backbone_weights].shape)))
                    print('%s-->%s'%(backbone_bias,self.size2str(pretrained_dict[backbone_bias].shape)))
                    
                    assert(model_dict[param_weights].shape==pretrained_dict[backbone_weights].shape)
                    assert(model_dict[param_bias].shape==pretrained_dict[backbone_bias].shape)
                    model_dict[param_weights]=pretrained_dict[backbone_weights]
                    model_dict[param_bias]=pretrained_dict[backbone_bias]
                offset+=th
        except:
            print ('loading model failed, {} may not exist'.format(model_path))
                    
    
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

    def apply_fix(self):
        for param in self.layer1.parameters():
            param.requires_grad=False
#        for param in self.layer2.parameters():
#            param.requires_grad=False
    
    def show_structure(self):
        print('Named parameters:')
        for k,v in self.named_parameters():
            print(k,v.shape)
        print('========\nState dict:')
        model_dict=self.state_dict()
        for k,v in model_dict.items():
            print(k,v.shape)
#        print('========\nModules:')