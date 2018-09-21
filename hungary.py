import numpy as np

MAX=100
lx=np.ones(MAX, dtype=np.float32)
ly=np.zeros(MAX, dtype=np.float32)              
sx=np.ones(MAX, dtype=np.bool)
sy=np.ones(MAX, dtype=np.bool)    

match=-1*np.ones(MAX, dtype=np.int32)

def pad(weights):
    rows, cols=weights.shape
    max_v=weights.max()
    if rows>cols:
        app_cols=max_v*np.ones((rows, rows-cols), dtype=weights.dtype)
        return np.hstack((weights, app_cols))
    elif rows<cols:
        app_rows=max_v*np.ones((cols-rows, cols), dtype=weights.dtype)
        return np.vstack((weights, app_rows))
    else:
        return weights

def dfs(u, weights):
    sx[u]=True;  
    for v in range(weights.shape[1]):
        if not sy[v] and lx[u]+ly[v] == weights[u,v]:
            sy[v]=True;  
            if match[v]==-1 or dfs(match[v], weights):
                match[v]=u;  
                return True;  
    return False;  

def hungary(all_dists, links):
    pass
    
def Kuhn_Munkres(weights):
    num_U=weights.shape[0]
    num_V=weights.shape[1]

    for i in range(num_U):
        lx[i]=weights[i].max()
    ly[:]=0
    match[:]=-1
    
    for u in range(num_U):
        while True:
            sx[:]=0
            sy[:]=0
            if dfs(u, weights):
                break
            
            inc=0x7fffffff
            for i in range(num_U):
                if sx[i]:
                    for j in range(num_V):
                        if not sy[j] and ((lx[i]+ly[j]-weights[i,j])<inc):  
                            inc=lx[i]+ly[j]-weights[i,j] ; 
            lx[sx==True]-=inc
            ly[sy==True]+=inc
            
    w_sum=0
    for i in range(num_U):
        if match[i]>=0:
            w_sum+=weights[match[i],i]
    return w_sum
            
if __name__=='__main__':
    all_dist=np.asarray([
            [5.3, 3.9, 8.6],
            [8.2, 3.2, 6.2],
            [8.3, 7.7, 3.1],
            [9.2, 3.8, 10.1]
            ])*10
    max_v=all_dist.max()
    all_dist=pad(all_dist)
    all_dist=max_v-all_dist
    all_dist=all_dist.astype(np.int32)
    print(all_dist)
    w_sum=Kuhn_Munkres(all_dist)
    print(w_sum)
    print(match[:all_dist.shape[0]])