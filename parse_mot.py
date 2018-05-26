# -*- coding: utf-8 -*-
"""
Created on Fri May 18 00:17:57 2018

@author: JYF
"""

import os
import os.path as op
import cv2

base_dir='I:/Develop/python/mot/'
data_file=op.join(base_dir,'MOT16-05.mp4')
gt_file=op.join(base_dir,'MOT16Labels/train/MOT16-05/gt/gt.txt')

def parse_label():
    label=open(gt_file,'r')
    
    lines=label.readlines()
    
    targets=[]
    
    cur_target_id=-1
    
    for line in lines:
        if len(line)<1:
            continue
        items=line.rstrip().split(',')
        items=list(map(float,items))
        target_id=int(items[1])
        frame_id=int(items[0])
        score=items[-1]
        if target_id!=cur_target_id:
            cur_target_id=target_id
            target=[]
            if cur_target_id!=-1:
                targets.append(target)
        t_entry={}
        t_entry['frame_id']=frame_id-1
        t_entry['bbox']=items[2:6]
        t_entry['score']=score
        target.append(t_entry)
    targets.append(target)
    print('Total targets: %d'%len(targets))
    
    label.close()
    return targets

def show_target(targets, index=1):
    cap=cv2.VideoCapture(data_file)
    target_index=index-1
    target=targets[target_index]
    total_frames=len(target)
    frame_start=target[0]['frame_id']
    
    fps=cap.get(cv2.cv.CV_CAP_PROP_FPS)
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_start)
    for i in range(total_frames):
        ok, image=cap.read()
        if not ok:
            break
        bbox=list(map(int,target[i]['bbox']))
        cv2.rectangle(image, (bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]), (255,0,0),2)
        cv2.putText(image, '%f'%target[i]['score'], (bbox[0],max(0,bbox[1]-15)), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,0,0))
        cv2.imshow('t', image)
#        if cv2.waitKey(int(1000.0/fps))==27:
        if cv2.waitKey()==27:
            break
    cv2.destroyAllWindows()


if __name__=='__main__':
    targets=parse_label()
    show_target(targets, index=5)
    
        