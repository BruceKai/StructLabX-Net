import torch
import torch.nn as nn

from utils.diceloss import BinaryTanimotoLoss 
from utils.focalloss import FocalLoss

        
class PartialLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super().__init__() 
        
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.focal_loss = FocalLoss(0.5)

    def forward(self,predict,target,loc):
        b,row,col = loc
        
        ## using focal loss
        # loss = self.focal_loss(predict[b,:,row,col],target[b,row,col].long())
        
        ## using cross entropy loss
        pred = torch.softmax(predict,dim=1)
        loss = self.ce_loss(pred[b,:,row,col],target[b,row,col].long())
        
        return loss

class JSDivLoss(torch.nn.Module):
    def __init__(self,reduction='mean'):
        super(JSDivLoss, self).__init__()
        self.kl_div = torch.nn.KLDivLoss(reduction=reduction)

    def forward(self, p, q):
        # Calculate M
        m = 0.5 * (p + q)
        
        # Compute JS divergence using KL divergence
        loss = 0.5 * (self.kl_div(p, m) + self.kl_div(q, m))
        
        return loss
    


class Loss(nn.Module):
    def __init__(self,reduction='mean',th_a =0.99,th_b=0.15):
        super(Loss, self).__init__()
        
        self.edge_loss = BinaryTanimotoLoss(reduction=reduction)
        self.seg_loss = PartialLoss(reduction=reduction)
        self.expansion_loss = PartialLoss(reduction=reduction)
        self.consistency_loss = nn.MSELoss(reduction=reduction)
        # self.consistency_loss = nn.KLDivLoss(reduction=reduction)
        # self.consistency_loss = JSDivLoss(reduction=reduction)
        
        self.lamda_edge = 5
        self.lamda_cons = 5
        
        self.th_a = th_a
        self.th_b = th_b
        
        self.value = []
    
    def forward(self,predict,target,edge_intensity):
        
        b,row,col = torch.where(target[:,1,:,:] == 1) # annotated pixels
        loss_seg = self.seg_loss(predict['seg'],target[:,0,:,:],[b,row,col]) 
        loss_edge = self.lamda_edge*self.edge_loss(predict['edge'],edge_intensity)

        pred_e = torch.softmax(predict['seg'],1)
        prob_e,_ = torch.max(pred_e,1)
        pred_e = torch.argmax(pred_e,1)

        pseudo = torch.zeros_like(pred_e).long()
        pseudo[b,row,col] = target[b,0,row,col].long()+1
        
        mask = prob_e > self.th_a 
        mask = mask & (edge_intensity<self.th_b)

        pseudo[mask] = pred_e[mask].long()+1

        b_e,row_e,col_e = torch.where(pseudo>0)
        loss_expan =  self.expansion_loss(predict['expansion'],pseudo-1,[b_e,row_e,col_e])

        loss_cons = self.lamda_cons*self.consistency_loss(torch.softmax(predict['expansion'],dim=1),torch.softmax(predict['seg'],dim=1))
        
        oa = (pred_e[b,row,col]==target[:,0,:,:][b,row,col]).sum()/torch.numel(row)

        loss = loss_seg + loss_expan + loss_edge + loss_cons 

 
        self.value = [
                    loss_seg.item(),
                    loss_expan.item(),
                    loss_edge.item(),
                    loss_cons.item(),
                    oa.detach().cpu().numpy(),
                    ]   
        return loss
            

    

