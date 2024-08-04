import torch
from torch import nn
from torch.nn import functional as F
from model.attention import ChannelAttention,SpatialAttention
# DownConvBlocks for SITS

class MultiTaskHead(nn.Module):
    def __init__(self,
                 num_classes,
                 filters
                 ):
        super(MultiTaskHead, self).__init__()
        
        self.seg_conv = OutConv(filters[0],num_classes)
        self.dropout = nn.Dropout(0.3)
        self.edge_conv = OutConv(filters[0],1)
        self.expand_conv = OutConv(filters[0],num_classes)
     
    def forward(self,*wargs):
        x = wargs[0]
    
        seg = self.seg_conv(x)
        # noise = torch.randn_like(x) * 0.1 + 0
        # x = x+noise
        expansion = self.expand_conv(self.dropout(x))
        edge = self.edge_conv(x)
        prediction = {
                      'seg':seg,
                      'edge':edge,
                      'expansion':expansion,
                      'feature':x
                      }
            
        return prediction



class DownConvBlocks(nn.Module):
    def __init__(self,in_channels,out_channels,k,s,p,
                 n_groups=4,norm_type='batch'):
        super().__init__()
        # self.norm_type = norm_type
        norm = dict({'batch':nn.BatchNorm2d,
                     'instance':nn.InstanceNorm2d,
                     'group':lambda num_feats: nn.GroupNorm(
                                num_channels=num_feats,
                                num_groups=n_groups)
                     })

        self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels, 
                                            kernel_size=k,padding=p,stride=s),
                                  norm[norm_type](in_channels),
                                  nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 
                                            kernel_size=3,padding=1,stride=1),
                                  norm[norm_type](out_channels),
                                  nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 
                                            kernel_size=3,padding=1,stride=1),
                                  norm[norm_type](out_channels),
                                  nn.ReLU(inplace=True))
    def forward(self,x):
        # b,t,c,h,w = x.shape
        # if self.norm_type == 'group':
        #     x = x.reshape(b*t,c,h,w)
        down = self.down(x)
        out = self.conv1(down)
        out = out+self.conv2(out)
        return out
    
class ASPPblock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,dilation,is_global):
        super(ASPPblock,self).__init__()
        padding = 0 if kernel_size==1 else dilation
        if is_global:
            self.aspp = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0),
                nn.ReLU(inplace=True) 
                )
        else:
            self.aspp = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                          dilation=dilation,padding=padding
                          ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) 
                )
    def forward(self,x):
        x = self.aspp(x)
        return x

class ASPP(nn.Module):
    def  __init__(self,in_channels,out_channels):
        super(ASPP,self).__init__()
        dilation = [1,2,4]
        self.aspp1 = ASPPblock(in_channels,out_channels,1,dilation[0],False)
        self.aspp2 = ASPPblock(in_channels,out_channels,3,dilation[1],False)
        self.aspp3 = ASPPblock(in_channels,out_channels,3,dilation[2],False)
        # self.aspp4 = ASPPblock(in_channels,out_channels,3,dilation[3],False)
        self.aspp4 = ASPPblock(in_channels,out_channels,1,dilation[0],True)
        
        self.conv = nn.Sequential(nn.Conv2d(out_channels*4, out_channels, 1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
    
    def forward(self,x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        # x5 = self.aspp5(x)
        x4 = F.interpolate(x4,(x.shape[-2],x.shape[-1]),mode='bilinear')
        out = self.conv(torch.cat([x1,x2,x3,x4],dim=1))
        return out

class ResBlock(nn.Module):
    # refinemen residual moduel
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels,output_channels,kernel_size=1,
                                            padding = 0),
                                  nn.BatchNorm2d(output_channels))
        self.res_conv = nn.Sequential(nn.Conv2d(output_channels,output_channels//2,kernel_size=3,
                                                 stride=1,padding=1,bias=True),
                                       nn.BatchNorm2d(output_channels//2),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(output_channels//2,output_channels,kernel_size=3,
                                                 stride=1, padding=1,bias=True),
                                       nn.BatchNorm2d(output_channels),
                                       )
        
    def forward(self,x):
        x = self.conv(x)
        # print(x.shape)
        res_x = self.res_conv(x)
        x = torch.relu(x+res_x)        
        return x   
 
class UpBlock(nn.Module):
    ''' refinement residual moduel
    '''
    def __init__(self,in_channels,out_channels,upsample_id):

        super().__init__()

        self.upsample_id = upsample_id
        if self.upsample_id == 1:
            self.cam = ASPP(in_channels,in_channels)
        else:
            self.scalefusion = JointAttention(in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.resblock = ResBlock(in_channels,out_channels)
    def forward(self,high_level,*kwargs):
        # x = self.conv(x)
        if (self.upsample_id>1)&(self.upsample_id<4):
            low_level = kwargs[0]
            x, attn = self.scalefusion(high_level,low_level)
            out = self.upsample(x)
            out = self.resblock(out)
            return out, attn
        elif self.upsample_id == 4:
            low_level = kwargs[0]
            x, attn = self.scalefusion(high_level,low_level)
            out = self.resblock(x)
            return out,attn
        else:
            x = self.cam(high_level)
            out = self.upsample(x)
            out = self.resblock(out)
            return out


class JointAttention(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        output_channels = input_channels
        self.channel_attn = ChannelAttention(input_channels*2,16)
        self.spatial_attn = SpatialAttention()
        self.fuse_conv = nn.Sequential(nn.Conv2d(input_channels*2,
                                                 output_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 bias=True),
                                       nn.BatchNorm2d(output_channels),
                                       nn.ReLU(inplace=True)) 

    def forward(self,X_c,X_f):
        x = torch.cat([X_c,X_f],dim=1)
        channel_attn = self.channel_attn(x)
        out = x*channel_attn
        spatial_attn = self.spatial_attn(x)
        out = out+x*spatial_attn
        out = self.fuse_conv(out)
        return out,[channel_attn,spatial_attn]

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        mid_channels = 32
        self.out_channels = out_channels
        self.conv = nn.Sequential(nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
                                nn.BatchNorm2d(mid_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1),
                                nn.BatchNorm2d(mid_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(mid_channels, out_channels, kernel_size=1),
                                      )

    def forward(self, x):
        
        # if self.out_channels > 1 :
        #     out = torch.softmax(self.conv(x),dim=1)
        # else:
        #     out = torch.sigmoid(self.conv(x))
        out = self.conv(x)
        return out

