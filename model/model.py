import torch
from torch import nn
from model.encoder import Encoder
                        
from model.decoder import Decoder
from model.model_parts import MultiTaskHead

class UTempoNet(nn.Module):
    def __init__(self,in_channels,num_classes,layer_num):
        super(UTempoNet,self).__init__()
        filters=[64,64,64,64]
        self.encoder = Encoder(in_channels,layer_num,True)
        self.decoder = Decoder()
        self.MTH = MultiTaskHead(num_classes=num_classes,filters=filters)


    def forward(self,*kwargs):

        temporal_data = kwargs[0]
        features,attns = self.encoder(temporal_data)
        out,_ = self.decoder(features)
        prediction = self.MTH(out)

        prediction.update({'attns':attns,})
        
        return prediction


        
        