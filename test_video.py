#!/usr/bin/python3

"""
Siamese fc network for tracking

@author: yfji
"""
from torch.autograd import Variable
import torch.optim as optim
import time
import os.path as op
import numpy as np
import numpy.random as nr
import cv2
from tracker import Tracker
from detector import Detector

yolo_root='/home/yfji/SourceCode/pytorch-yolo-v3/'

def find_max(preds):
    ws=preds[:,3]-preds[:,1]
    hs=preds[:,4]-preds[:,2]
    areas=ws*hs
    return np.argmax(areas)
    
def track_largest():
    cap=cv2.VideoCapture('MOT16-06.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)  
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    assert cap.isOpened(), 'invalid capture'
    
    wcap=cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    det=Detector(model_def_file=op.join(yolo_root,'cfg/yolov3.cfg'), weights_file=op.join(yolo_root,'yolov3.weights'))
    track=Tracker()
    sz=track.net_input_size
    first=1
    max_box=None
    t_box=None
    
    start = time.time()
    
    frames=0
    display=10
    while True:
        ok,frame=cap.read()
        if not ok:
            break
        
        output=det.detect(frame)
        if first:
            max_box=output[find_max(output),1:5].astype(np.int32)
            t_box=max_box
            first=0
        else:
            x1=np.zeros((output.shape[0],3,sz,sz),dtype=np.float32)
            x2=np.zeros((output.shape[0],3,sz,sz),dtype=np.float32)
            
            target=frame[max_box[1]:max_box[3],max_box[0]:max_box[2],:]
            target=cv2.resize(target, (sz,sz), interpolation=cv2.INTER_LINEAR).transpose(2,0,1).astype(np.float32)
            target-=target.min();target/=target.max()
            for i in range(output.shape[0]):
                x1[i]=target
                query=frame[int(output[i,2]):int(output[i,4]),int(output[i,1]):int(output[i,3]),:]
                query=cv2.resize(query, (sz,sz), interpolation=cv2.INTER_LINEAR).transpose(2,0,1).astype(np.float32)
                query-=query.min();query/=query.max()
                x2[i]=query
            
            dist_l1=track.run_data(x1,x2)
            dist_l1=dist_l1.data.cpu().numpy()
            min_ind=np.argmin(dist_l1)
            t_box=output[min_ind,1:5]
        cv2.rectangle(frame, (int(t_box[0]),int(t_box[1])),(int(t_box[2]),int(t_box[3])), det.colors[0], 2)
#        cv2.imshow('', frame)
        wcap.write(frame)
        frames+=1
        if frames%display==0:
            print("FPS of the video is {:5.2f}, dist is {:5.2f}".format( frames / (time.time() - start), dist_l1.min()))
            if dist_l1.min()>10:
                max_box=output[find_max(output),1:5].astype(np.int32)
#        if cv2.waitKey(2)==27:
#            break

#    cv2.destroyAllWindows()    
    
def track_mot():
    pass

if __name__=='__main__':
    track_largest()

