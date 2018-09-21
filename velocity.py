import numpy as np
from collections import deque

class Predictor(object):
    def __init__(self):
        pass
    
    def _update(self, new_pos):
        pass
    
    def _reset(self):
        pass
    
class SimpleFilter(Predictor):
    def __init__(self):
        super(Predictor, self).__init__()
        self.memory=10
        self.history=deque(maxlen=self.memory)
        
    def _update(self, new_pos):
        if len(self.history)<=1:
            self.history.append(new_pos)
            return 0,0
        x=new_pos[0]
        y=new_pos[1]
        self.history.append(new_pos)
        speed_x=0
        speed_y=0
        for i in range(1,len(self.history)):
            pt1=self.history[i-1]
            pt2=self.history[i]
            speed_x+=1.0*(pt2[0]-pt1[0])
            speed_y+=1.0*(pt2[1]-pt1[1])  
        spd_x_one_time=1.0*(x-self.history[0][0])
        spd_y_one_time=1.0*(y-self.history[0][1])
        speed_x/=(len(self.history)-1)
        speed_y/=(len(self.history)-1)
        spd_x_one_time/=(len(self.history)-1)
        spd_y_one_time/=(len(self.history)-1)
        
        spd_x=(speed_x+spd_x_one_time)/2
        spd_y=(speed_y+spd_y_one_time)/2
        
        return spd_x, spd_y
    
    def _reset(self):
        self.history=deque(maxlen=self.memory)
        
        
    
class KalmanFilter(Predictor):
    def __init__(self):
        pass
    
    def _update(self, new_pos):
        pass
    
    def _reset(self):
        pass

class Velocity(object):
    def __init__(self):
        self.filter=SimpleFilter()
        self.reset()
    
    def update(self, new_pos):
        self.cur_pos=new_pos
        self.spd_x, self.spd_y=self.filter._update(new_pos)
        if self.spd_x==0 and self.spd_y==0:
            self.direction=0
        else:
            theta=np.arctan(self.spd_y/self.spd_x)
            if self.spd_y>0 and self.spd_x>0:
                self.direction=theta
            elif self.spd_y>0 and self.spd_x<0:
                self.direction=np.pi-theta
            elif self.spd_y<0 and self.spd_x<0:
                self.direction=np.pi+theta
            else:
                self.direction=2*np.pi-theta
    
    def calc_distance(self, new_pos):
        dist_x=new_pos[0]-self.cur_pos[0]
        dist_y=new_pos[1]-self.cur_pos[1]
        return np.sqrt(dist_x**2+dist_y**2)
        
    def reset(self):
        self.spd_x=0
        self.spd_y=0
        self.direction=0    #0-2pi
        self.cur_pos=None
        
        self.filter._reset()
    
    def predict(self):
        next_x=self.cur_pos[0]+self.spd_x
        next_y=self.cur_pos[1]+self.spd_y
        return (next_x,next_y)