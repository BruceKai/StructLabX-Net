import torch
import torch.nn as nn
from torchvision import models
from model.model_parts import DownConvBlocks,ASPP
from torch.nn import functional as F
# resnet=models.resnet50(pretrained=True)
# resnet=models.resnet18(pretrained=True)

class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 num_layers,
                 bidirectional=True,
                 **kwargs):
        super().__init__()
        filters=[64,128,256,512]

        self.input_conv = nn.Sequential(
                                        nn.Conv2d(in_channels,filters[0],3,1,1),
                                        nn.BatchNorm2d(filters[0]),
                                        nn.ReLU(inplace=True))

        self.layer1 = DownConvBlocks(filters[0],filters[1],
                                     k=3,s=2,p=1)
        self.layer2 = DownConvBlocks(filters[1],filters[2],
                                     k=3,s=2,p=1)
        self.layer3 = DownConvBlocks(filters[2],filters[3],
                                     k=3,s=2,p=1)
        # self.layer4 = DownConvBlocks(filters[3],filters[4],
        #                              k=3,s=2,p=1)
        
        self.conv = nn.ModuleList([nn.Sequential(
                                            nn.Conv2d(filters[i],64,3,
                                                stride=1,padding=1),
                                            nn.ReLU(inplace=True)) 
                                  for i in range(4)])
        
        # temporal encoder atbilstm
        self.temporal_encoder = nn.ModuleList([AtBiLSTM(filters[0],
                                                        filters[0]//2,
                                                        dropout=0.5,
                                                        num_layers=num_layers,
                                                        bidirectional=bidirectional)
                                               for i in range(4)])
        

        
    def forward(self,x):
        # x shapes: (_,12,3,64,64)
        b,t,c,h,w = x.shape
        x = x.reshape(b*t,c,h,w)
        

        level1 = self.input_conv(x)
        level2 = self.layer1(level1) # shapes: (_,128,32,32) 
        level3 = self.layer2(level2) # shapes: (_,256,16,16)
        level4 = self.layer3(level3) # shapes: (_,512,8,8) 
        # level5 = self.layer4(level4)
        
        
        features = [level1,level2,level3,level4]
        attns = []
        for i in range(4):
            features[i] = self.conv[i](features[i])
            _,c,h,w = features[i].shape 
            features[i] = features[i].reshape(b,t,c,h,w).permute(0,3,4,1,2).reshape(b*h*w,t,c)
            features[i],attn = self.temporal_encoder[i](features[i],[b,t,c,h,w]) # shapes: (12,64,64,64)
            attns.append(attn)

        return features,attns

class AtBiLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 bidirectional=True,
                 dropout=0.5,
                 **kwargs):
        super().__init__()

        self.temporal_encoder = nn.LSTM(
                                        input_size= input_size,
                                        hidden_size= hidden_size,
                                        num_layers=num_layers,
                                        batch_first = True,
                                        bidirectional = bidirectional,
                                        dropout=dropout,
                                        )              
        
        self.temporal_aggregator = TemporalAggregator(hidden_size,bidirectional) 


        
    def forward(self,x,shapes):
        
        b,t,c,h,w = shapes
        te,_ = self.temporal_encoder(x) # shapes: (12,64,64,64)
        te = te.reshape(b,h,w,t,c).permute(0,3,4,1,2)
        features,attn = self.temporal_aggregator(te) # shapes: (64,64,64)

        return features,attn
    
class TemporalAggregator(nn.Module):
    def __init__(self,hidden_size,bidirectional=True):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels=(1*bidirectional+1)*hidden_size,
                                    out_channels=1,kernel_size=1,
                                    stride=1,padding=0,bias=True),
                                  nn.BatchNorm2d(1))
                  
    def forward(self,x):

        b,t,c,h,w = x.shape 
        temporal_attn = self.conv(x.reshape(b*t,c,h,w))
        temporal_attn = torch.softmax(temporal_attn.reshape(b,t,1,h,w),dim=1)
        weighted_out = torch.sum(temporal_attn*x,dim=1)
            
        return weighted_out,temporal_attn