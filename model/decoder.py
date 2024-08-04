import torch.nn as nn
from model.model_parts import UpBlock,OutConv
import torch
from torch.nn import functional as F
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # filters=[64,128,256,512,1024]
        filters = [64,64,64,64,64]
        self.filters = filters
        self.upblock_1 = UpBlock(filters[4],filters[3],1)
        self.upblock_2 = UpBlock(filters[3],filters[2],2)
        self.upblock_3 = UpBlock(filters[2],filters[1],3)
        self.upblock_4 = UpBlock(filters[1],filters[0],4)

  
    def forward(self,*wargs):

        temporal = wargs[0]
        tm_l2,tm_l3,tm_l4,tm_l5 = temporal
        out = self.upblock_1(tm_l5) # shapes: (256,16,16)
        out,sa1 = self.upblock_2(out,tm_l4) # shapes: (128,32,32)
        out,sa2 = self.upblock_3(out,tm_l3) # shapes: (64,64,64)
        out,sa3 = self.upblock_4(out,tm_l2) # shapes: (64,64,64)

        return out,[sa1,sa2,sa3]


