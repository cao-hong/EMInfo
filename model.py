import torch
import torch.nn as nn

from frn import FilterResponseNorm3d

class conv_block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, residual=False, identity=False, dropout=False):
        super(conv_block, self).__init__()

        self.identity = identity
        self.residual = residual
        self.dropout = dropout

        if identity and mid_ch == out_ch:
            self.frnI = FilterResponseNorm3d(out_ch)
                
        if residual:
            self.convR = nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, bias=True)
            self.frnR = FilterResponseNorm3d(out_ch)
        
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.frn1 = FilterResponseNorm3d(mid_ch)
        self.drop1 = nn.Dropout(0.1, inplace=True)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.frn2 = FilterResponseNorm3d(out_ch)
        self.drop2 = nn.Dropout(0.1, inplace=True)

    def forward(self, x):

        if self.residual:
            xr = self.convR(x)
            xr = self.frnR(x)
        else:
            xr = .0

        x = self.conv1(x)
        x = self.frn1(x)
        x = self.drop1(x) if self.dropout else x
        
        if self.identity:
            xi =  self.frnI(x)
        else:
            xi = .0

        x = self.conv2(x)
        x = self.frn2(x)
        x = self.drop2(x) if self.dropout else x

        return x + xr + xi

class NestedUNet(nn.Module):
    '''
        NestedUNet
        Implementation of paper "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"(https://arxiv.org/pdf/1807.10165.pdf)
        https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py
    '''
    def __init__(self, in_channels=1, init_channels=32, growth_rate=2, n_classes=4, dropout=False):
        super(NestedUNet, self).__init__()

        depth = 4

        n = init_channels
        filters = [n * growth_rate ** i for i in range(depth)]

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dropout = dropout

        # depth 1
        self.conv0_0 = conv_block(in_channels, filters[0], filters[0], dropout=self.dropout) # in,     depth 1
    
        # depth 2
        self.conv1_0 = conv_block(filters[0], filters[1], filters[1], dropout=self.dropout)  # down 1, depth 2
        self.conv0_1 = conv_block(filters[0] + filters[1], filters[0], filters[0], dropout=self.dropout) # up 1, depth 2

        # depth 3
        self.conv2_0 = conv_block(filters[1], filters[2], filters[2], dropout=self.dropout)  # down 2, depth 3
        self.conv1_1 = conv_block(filters[1] + filters[2], filters[1], filters[1], dropout=self.dropout) # up 1, depth 3
        self.conv0_2 = conv_block(filters[0]*2 + filters[1], filters[0], filters[0], dropout=self.dropout) # up 2, depth 3

        # depth 4
        self.conv3_0 = conv_block(filters[2], filters[3], filters[3])  # down 3, depth 4
        self.conv2_1 = conv_block(filters[2] + filters[3], filters[2], filters[2], dropout=self.dropout) # up 1, depth 4
        self.conv1_2 = conv_block(filters[1]*2 + filters[2], filters[1], filters[1], dropout=self.dropout) # up 2, depth 4
        self.conv0_3 = conv_block(filters[0]*3 + filters[1], filters[0], filters[0], dropout=self.dropout) # up 3, depth 4

        self.final = nn.Conv3d(filters[0], n_classes, kernel_size=1, padding=0, bias=True) # out

        for m in self.modules():
            if torch.jit.isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                   torch.nn.init.constant_(m.bias.data,0.3)


    def forward(self, x):
        # depth 1
        x0_0 = self.conv0_0(x)

        # depth 2
        x1_0 = self.conv1_0(self.pool(x0_0)) # down 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1)) # up 1

        # depth 3
        x2_0 = self.conv2_0(self.pool(x1_0)) # down 2
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1)) # up 1
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1)) # up 2

        # depth 4
        x3_0 = self.conv3_0(self.pool(x2_0)) # down 3
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1)) # up 1
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1)) # up 2
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1)) # up 3

        output = self.final(x0_3)
        return output

