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
from velocity import Velocity
import hungary as H

yolo_root='/home/yfji/SourceCode/pytorch-yolo-v3/'

def find_max(preds):
    ws=preds[:,3]-preds[:,1]
    hs=preds[:,4]-preds[:,2]
    areas=ws*hs
    return np.argmax(areas)

def calc_center(image, t_box):
    box=t_box.astype(np.int32)
    patch=cv2.cvtColor(image[box[1]:box[3], box[0]:box[2],:], cv2.COLOR_BGR2GRAY).astype(np.float32)
    #0,1 moments
    patch-=patch.min()
    patch/=patch.max()
    
    coeff_mat10=np.zeros((patch.shape[0],0), dtype=np.float32)
    for i in range(patch.shape[1]):
        coeff_mat10=np.hstack((coeff_mat10, (i+1)*np.ones((patch.shape[0],1)))) 
        
    coeff_mat01=np.zeros((0,patch.shape[1]), dtype=np.float32)
    for i in range(patch.shape[0]):
        coeff_mat01=np.vstack((coeff_mat01, (i+1)*np.ones((1,patch.shape[1]))))
    
    M00=np.sum(patch)
    M10=np.sum(np.multiply(patch, coeff_mat10))
    M01=np.sum(np.multiply(patch, coeff_mat01))
    
    xc=M10/M00
    yc=M01/M00
    
    xc+=box[0]
    yc+=box[1]
    return (xc,yc)
    
def track_largest():
    cap=cv2.VideoCapture('MOT17-04.avi')
    fps = cap.get(cv2.CAP_PROP_FPS)  
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    assert cap.isOpened(), 'invalid capture'
    
    evaluator=Velocity()
    wcap=cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    det=Detector(model_def_file=op.join(yolo_root,'cfg/yolov3.cfg'), weights_file=op.join(yolo_root,'yolov3.weights'))
    track=Tracker()
    
    sz=track.net_input_size
    first=1
    max_box=None
    t_box=None
    last_t_box=None
    next_center=None
    moment_center=None
    frame=None
    last_frame=None
    
    start = time.time()
    
    frames=0
    failed_frames=0
    display=20
    min_dist=0
    
    sim_thresh=28
    dist_thresh=20
    pad_thresh=10
    
    while True:
        last_frame=frame
        ok,frame=cap.read()
        if not ok:
            break
        if frames==500:
            break
        output=det.detect(frame)
        
        if first:
            max_box=output[find_max(output),1:5].astype(np.int32)
            t_box=max_box
            moment_center=calc_center(frame, t_box)
            evaluator.update(moment_center)
            first=0
        else:
            boxes=output[:,1:5].astype(np.int32)
            x1=np.zeros((boxes.shape[0],3,sz,sz),dtype=np.float32)
            x2=np.zeros((boxes.shape[0],3,sz,sz),dtype=np.float32)
            
            target=last_frame[t_box[1]:t_box[3],t_box[0]:t_box[2],:]

            target=cv2.resize(target, (sz,sz), interpolation=cv2.INTER_LINEAR).transpose(2,0,1).astype(np.float32)
            target-=target.min();target/=target.max()
            
            for i in range(boxes.shape[0]):
                x1[i]=target[...]
                query=frame[boxes[i,1]:boxes[i,3],boxes[i,0]:boxes[i,2],:]
                query=cv2.resize(query, (sz,sz), interpolation=cv2.INTER_LINEAR).transpose(2,0,1).astype(np.float32)
                query-=query.min();query/=query.max()
                x2[i]=query[...]
            
            dist_l1=track.run_data(x1,x2)
            
            dist_l1=dist_l1.data.cpu().numpy()
            min_ind=np.argmin(dist_l1)
            min_dist=dist_l1[min_ind]
            
            t_box=output[min_ind,1:5].astype(np.int32)
            moment_center=calc_center(frame, t_box)
            dist=evaluator.calc_distance(moment_center)
            next_center=evaluator.predict()

            if min_dist<sim_thresh:
#                print('Successful siamese match')
                evaluator.update(moment_center)
                failed_frames=0
            else:
                if failed_frames==0 or (next_center[0]<pad_thresh or next_center[1]<pad_thresh or next_center[0]>frame.shape[1]-pad_thresh or next_center[1]>frame.shape[0]-pad_thresh):  
                    #out of image, find a new target to track
                    max_box=output[find_max(output),1:5].astype(np.int32)
                    t_box=max_box
                    evaluator.reset()
                    moment_center=calc_center(frame, t_box)
                    evaluator.update(moment_center)
                    next_center=None
                    failed_frames=0
                else:   #update using motion model
                    delta_x=int(next_center[0]-evaluator.cur_pos[0])
                    delta_y=int(next_center[1]-evaluator.cur_pos[1])
#                    print('Update by motion model: %f, %f'%(delta_x, delta_y))
                    t_box=last_t_box
                    if delta_x>0:
                        t_box[2]+=delta_x
                    else:
                        t_box[0]+=delta_x
                    if delta_y>0:
                        t_box[3]+=delta_y
                    else:
                        t_box[1]+=delta_y  
                    t_box[[0,2]]=np.minimum(frame.shape[1],np.maximum(0,t_box[[0,2]]))
                    t_box[[1,3]]=np.minimum(frame.shape[0],np.maximum(0,t_box[[1,3]]))
                    evaluator.update(next_center)
                    failed_frames+=1
        
        cv2.rectangle(frame, (int(t_box[0]),int(t_box[1])),(int(t_box[2]),int(t_box[3])), (0,255,0), 2)
        if next_center is not None:
            cv2.circle(frame, (int(next_center[0]),int(next_center[1])), 5, (0,0,255), -1)
#        cv2.imshow('', frame)
        wcap.write(frame)
        frames+=1
        if frames%display==0:
            print("FPS of the video is {:5.2f}, min_dist is {:5.2f}, max_dist is {:5.2f}, average_dist is {:5.2f}".format( frames / (time.time() - start), dist_l1.min(), dist_l1.max(), dist_l1.mean()))
#            if dist_l1.min()>13:
#                max_box=output[find_max(output),1:5].astype(np.int32)
#        if cv2.waitKey(2)==27:
#            break
#    cv2.destroyAllWindows()    
    
def track_mot():
    cap=cv2.VideoCapture('MOT16-06.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)  
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    assert cap.isOpened(), 'invalid capture'
    
    wcap=cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    
    det=Detector(model_def_file=op.join(yolo_root,'cfg/yolov3.cfg'), weights_file=op.join(yolo_root,'yolov3.weights'))
    m_tracker=Tracker()
    
    sz=m_tracker.net_input_size
    first=1
    
    t_boxes=None
    valid_boxes=None
    valid_inds=None
    
    colors=det.colors
    states=np.zeros(1000, dtype=np.int32)
    
    start = time.time()
    
    frames=0
    display=1
    
    U=0
    V=0
    M=0
    distance=None
    matches=None
    
    frame=None
    last_frame=None
    faster=1    #discard lost targets or not
    
    while True:
        last_frame=frame
        ok,frame=cap.read()
        if not ok:
            break
        if frames==20:
            break
        output=det.detect(frame)
        
        if first:
            t_boxes=output[:,1:5].astype(np.int32)
            U,V=t_boxes.shape[0],t_boxes.shape[0]
            states[:U]=1
            first=0
        else:   
            boxes=output[:,1:5].astype(np.int32)
            U,V=boxes.shape[0],t_boxes.shape[0]
            
            valid_boxes=t_boxes[states[:V]==1] if faster else t_boxes
            valid_inds=np.arange(V)[states[:V]==1] if faster else np.arange(V)
            
            M=valid_boxes.shape[0]
            
            x1=np.zeros((M*U,3,sz,sz),dtype=np.float32)
            x2=np.zeros((M*U,3,sz,sz),dtype=np.float32)
            for i in range(U):
                target=frame[boxes[i,1]:boxes[i,3],boxes[i,0]:boxes[i,2],:]
                target=cv2.resize(target, (sz,sz), interpolation=cv2.INTER_LINEAR).transpose(2,0,1).astype(np.float32)
                target-=target.min();target/=target.max()
                for j in range(M):
                    x1[i*M+j]=target
                    query=last_frame[valid_boxes[j,1]:valid_boxes[j,3],valid_boxes[j,0]:valid_boxes[j,2],:]
                    query=cv2.resize(query, (sz,sz), interpolation=cv2.INTER_LINEAR).transpose(2,0,1).astype(np.float32)
                    query-=query.min();query/=query.max()
                    x2[i*M+j]=query
                    
            dist_l1=m_tracker.run_data(x1,x2).data.cpu().numpy() 
            
            dist_l1=dist_l1.reshape(U,M)
            
            dist_l1=H.pad(dist_l1)*100
            dist_int=dist_l1.astype(np.int32)
            distance=dist_int
            dist_int=dist_int.max()-dist_int
            
            H.Kuhn_Munkres(dist_int)
            
            matches=H.match[:U]
            
            states[:U]=0
            for i in range(U):
                j=matches[i]
                if j<M:               #cur target < last target, some targets lost
                    t_boxes[valid_inds[j]]=boxes[i]
                    states[valid_inds[j]]=1
            for i in range(U):
                j=matches[i]
                if j>=M:    #cur target > last target, new targets appear
                    found=0
                    for k in range(V):
                        if states[k]==0:
                            t_boxes[k]=boxes[i]
                            found=1
                            states[k]=1
                            break
                    if not found:
                        t_boxes=np.vstack((t_boxes, boxes[i]))
                        states[j]=1
                
        for i in range(t_boxes.shape[0]):
            if states[i]:
                cv2.rectangle(frame, (int(t_boxes[i,0]),int(t_boxes[i,1])),(int(t_boxes[i,2]),int(t_boxes[i,3])), colors[i], 2)
        wcap.write(frame)
        frames+=1
        if frames%display==0:
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            if distance is not None:
                print(distance[:U,:V])
                print(matches[:t_boxes.shape[0]])
            print(U,V,M)

#def track_iou():
    

if __name__=='__main__':
    track_largest()

