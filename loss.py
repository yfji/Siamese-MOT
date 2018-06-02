import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastLoss(nn.Module):
    def __init__(self, margin=10):
        super(ContrastLoss, self).__init__()
        self.margin = margin
        
    def forward(self, xp, xn, y, average=True):
        distances = (xp - xn).pow(2).sum(1)  # squared distances
        losses = 0.5 * (y.float() * distances +
                        (1 + -1 * y).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if average else losses.sum()

       
class TripletLoss(nn.Module):
    def __init__(self, margin=10):
        super(TripletLoss, self).__init__()
        self.margin=margin
    
    def forward(self, anchor, xp, xn, average=True):
        distance_positive = (anchor - xp).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - xn).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin=10):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.index=0
        
    def forward(self, xps, xns, labels):
#        if embeddings.is_cuda:
#            positive_pairs = positive_pairs.cuda()
#            negative_pairs = negative_pairs.cuda()
        '''
        pairs: [b,2,dim]
        labels: b
        '''
        xps_vec=xps.view(xps.size()[0],-1)
        xns_vec=xns.view(xns.size()[0],-1)
        dist_l2=torch.sum(torch.pow(xps_vec-xns_vec,2),1)
        dist=torch.sqrt(dist_l2)
#        print('dist: %f'%dist.mean())
        positive_loss = 0.5*labels*dist_l2
        negative_loss = 0.5*(1.0-labels)*torch.pow(F.relu(1.0*self.margin - dist),2)
#        negative_loss=0.5*(1.0-labels)*torch.pow(torch.clamp(1.0*self.margin-dist, min=0.0), 2)
#        if self.index==20:
#            print('pos loss',positive_loss.mean())
#            print('neg loss',negative_loss.mean())
#            print(xps_vec[0,:16])
#            print(xns_vec[0,:16])
#            print('dist: %f'%dist.mean())
#            self.index=0
#        self.index+=1
        loss=positive_loss+negative_loss
        return loss.mean(), dist


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """
    def __init__(self, margin=10):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchors, xps, xns):
#        if embeddings.is_cuda:
#            triplets = triplets.cuda()
        '''
        triplets: [b,3,dim]
        '''
        ap_distances = torch.sum(torch.pow(anchors - xps, 2), 1)  # .pow(.5)
        an_distances = torch.sum(torch.pow(anchors - xns, 2), 1)  # .pow(.5)
        
        loss = F.relu(ap_distances - an_distances + self.margin)

        return loss.mean(), ap_distances.mean(), an_distances.mean()