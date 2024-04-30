
import torch
import torch.nn as nn


class modelFrownetSimple(nn.Module):
    '''
        A model img to img. We follow the paper FlowNet: Learning Optical Flow with Convolutional Networks
    '''

    def __init__(self, HEIGH : int = 384, WIDTH : int = 512, in_CHANNELS : int = 6): #todo 6 ? 
        super(modelFrownetSimple).__init__()

        self.Conv1   = self.block_Conv(in_channels = in_CHANNELS, out_channels = 64, kernel_size = 7, stride = 1)
        self.Conv2   = self.block_Conv(in_channels = 64  , out_channels = 128 , kernel_size = 5, stride = 2)
        self.Conv3   = self.block_Conv(in_channels = 128 , out_channels = 256 , kernel_size = 5, stride = 2)
        self.Conv3_1 = self.block_Conv(in_channels = 256 , out_channels = 256 , kernel_size = 3, stride = 1)
        self.Conv4   = self.block_Conv(in_channels = 256 , out_channels = 512 , kernel_size = 3, stride = 2)
        self.Conv4_1 = self.block_Conv(in_channels = 512 , out_channels = 512 , kernel_size = 3, stride = 1)
        self.Conv5   = self.block_Conv(in_channels = 512 , out_channels = 512 , kernel_size = 3, stride = 2)
        self.Conv5_1 = self.block_Conv(in_channels = 512 , out_channels = 512 , kernel_size = 3, stride = 1)
        self.Conv6   = self.block_Conv(in_channels = 512 , out_channels = 1024, kernel_size = 3, stride = 2)
        self.Conv6_1 = self.block_Conv(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 2)
        #todo why do we need Conv6_1, only for loos6 ??

        #TODO
        #* Refinament block
        self.flow6   = self.block_Flow(in_channels = 1024)
        self.flow5   = self.block_Flow()
        self.deconv5 = self.block_Deconv()
        self.flow5   = self.block_Flow()
        self.deconv4 = self.block_Deconv()
        self.flow4   = self.block_Flow()
        self.deconv3 = self.block_Deconv()
        self.flow3   = self.block_Flow()
        self.deconv2 = self.block_Deconv()

    def forward(self, input):
        #TODO
        return input
    
    #todo def block_Refinement(self):


    def block_Conv(self, 
                in_channels  : int, 
                out_channels : int, 
                kernel_size  : int, 
                stride : int):

        return nn.Sequential(
            nn.Conv2d(
                in_channels  = in_channels, 
                out_channels = out_channels, 
                kernel_size  = kernel_size, 
                stride  = stride, 
                padding = kernel_size//2
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace = True)
        )
    
    def block_Deconv(self, 
                    in_channels  : int, 
                    out_channels : int, 
                    kernel_size  : int, 
                    stride  : int, 
                    padding : int
                    ):

        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels  = in_channels,
                out_channels = out_channels, 
                kernel_size  = kernel_size, 
                stride  = stride, 
                padding = padding
                ), 
            nn.BatchNorm2d(out_channels),  
            nn.SiLU()
        )

    def block_Flow(self, in_channels : int):
        return nn.Conv2d(in_channels, 2, kernel_size = 3, stride = 1, padding = 1)


    def __str__(self):
        print("Summary of model ")
        return "model sumarry"
    


