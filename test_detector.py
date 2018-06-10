# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 14:08:43 2018

@author: JYF
"""

from detector import Detector
import cv2
import numpy as np
import random

if __name__=='__main__':
    cap=cv2.VideoCapture('MOT16-06.mp4')
    assert cap.isOpened(), 'invalid capture'
    
    det=Detector(model_def_file='cfg/yolov3.cfg', weights_file='yolov3.weights')
    
    while True:
        ok,frame=cap.read()
        if not ok:
            break
        
        output=det.detect(frame)
#        print(output.shape[0])
        for x in output:
            cv2.rectangle(frame, (int(x[1]),int(x[2])),(int(x[3]),int(x[4])), random.choice(det.colors), 2)
        
        cv2.imshow('', frame)
        if cv2.waitKey(2)==27:
            break
    cv2.destroyAllWindows()