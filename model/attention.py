import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Softmax


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(2,1,kernel_size=7,
                                          stride=1,padding=3,bias=True),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid())
    def forward(self,x):
        avg_ = torch.mean(x,dim=1,keepdim=True)
        max_,_ = torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avg_,max_],dim=1)
        spatial_attn = self.conv(x)
        return spatial_attn

class ChannelAttention(nn.Module):
    def __init__(self,input_channels,reduction):
        super().__init__()
        output_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_attn_block=nn.Sequential( 
                                        nn.Conv2d(input_channels,input_channels//reduction,
                                            kernel_size=(1,1),stride=1,padding=0,bias=True),
                                        # nn.BatchNorm2d(input_channels//reduction),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(input_channels//reduction,output_channels,
                                               kernel_size=(1,1),stride=1,padding=0,bias=True),
                                        # nn.BatchNorm2d(input_channels),
                                        )
        
    def forward(self,x):
        max_pool_re = self.max_pool(x)
        avg_pool_re = self.avg_pool(x)           
        max_channel_attn = self.channel_attn_block(max_pool_re)
        avg_channel_attn = self.channel_attn_block(avg_pool_re)
        channel_attn = torch.sigmoid(max_channel_attn+avg_channel_attn)
        
        return channel_attn
