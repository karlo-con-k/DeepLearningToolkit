
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class model_U_Net(nn.Module):

    '''
        Implementation of the model U-Net for img segmentation

        Parameters
        ==========

        HEIGHT : int
            Height of the img
        WIDTH : int
            WIDTH of the img
        in_CHANNELS : int
            Channels of the img (default 1, because we use black img)

    '''

    def __init__(self, 
                HEIGHT : int = 572,
                WIDTH  : int = 572,
                in_CHANNELS: int = 1):
        super(model_U_Net, self).__init__()

        
        self.conv1 = self.conv_block(in_CHANNELS , 64, 3, 1) #TODO add double conv in the conv block for les layers
        self.conv2 = self.conv_block(64, 64, 3, 1)

        self.maxPol1 = nn.MaxPool2d(kernel_size = 2)

        self.conv3 = self.conv_block(64 , 128, 3, 1)
        self.conv4 = self.conv_block(128, 128, 3, 1)

        self.maxPol2 = nn.MaxPool2d(kernel_size = 2)

        self.conv5 = self.conv_block(128, 256, 3, 1)
        self.conv6 = self.conv_block(256, 256, 3, 1)

        self.maxPol3 = nn.MaxPool2d(kernel_size = 2)

        self.conv7 = self.conv_block(256, 512, 3, 1)
        self.conv8 = self.conv_block(512, 512, 3, 1)

        self.maxPol4 = nn.MaxPool2d(kernel_size = 2)

        self.conv9  = self.conv_block(512 , 1024, 3, 1)
        self.conv10 = self.conv_block(1024, 1024, 3, 1)

        self.upConv1 = self.convUp_block(1024, 512, 3, 2) #* pixshuffle ? 

        self.conv11   = self.conv_block(1024, 512, 3, 1)
        self.conv11_1 = self.conv_block(512 , 512, 3, 1)

        self.upConv2 = self.convUp_block(512, 256, 3, 2)

        self.conv12    = self.conv_block(512, 256, 3, 1) 
        self.conv12_1  = self.conv_block(256, 256, 3, 1)
        
        self.upConv3 = self.convUp_block(256, 128, 3, 2)

        self.conv13   = self.conv_block(256, 128, 3, 1)
        self.conv13_1 = self.conv_block(128, 128, 3, 1)

        self.upConv4 = self.convUp_block(128, 64, 3 ,2)

        self.conv14   = self.conv_block(128, 64, 3, 1)
        self.conv14_1 = self.conv_block(64, 64, 3, 1)

        self.lastConv = self.conv_block(64, 2, 1, 1)


    def conv_block(self, 
                in_channels  : int,
                out_channels : int, 
                kernel_size  : int, 
                stride : int):

        return nn.Sequential(
            nn.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = kernel_size,
                    stride = stride
                    ), 
            nn.SiLU() #TODO use SilU
        )

    def convUp_block(self, 
                in_channels  : int,
                out_channels : int, 
                kernel_size  : int, 
                stride : int):

        return nn.Sequential(
            nn.ConvTranspose2d(
                            in_channels  = in_channels, 
                            out_channels = out_channels, 
                            kernel_size  = kernel_size, 
                            stride  = stride,
                            padding = 1,
                            output_padding = 1
                            ),
            nn.ReLU()#TODO use SilU
        )

    def forward(self, inPut):

        outPut = self.conv1(inPut)
        outPut = self.conv2(outPut)
        copy1  = transforms.Resize((392, 392))(outPut)

        outPut = self.maxPol1(outPut) 

        outPut = self.conv3(outPut)
        outPut = self.conv4(outPut)
        copy2  = outPut[:, 40:240, 40:240]

        outPut = self.maxPol2(outPut)

        outPut = self.conv5(outPut)
        outPut = self.conv6(outPut)
        copy3  = outPut[:, 16:120, 16:120 ] #* must be 104x104x512
        outPut = self.maxPol3(outPut)

        outPut = self.conv7(outPut)
        outPut = self.conv8(outPut)
        copy4  = outPut[:, 4:60, 4:60] #todo maxPol ? 

        outPut = self.maxPol4(outPut)

        outPut = self.conv9(outPut)
        outPut = self.conv10(outPut)

        outPut = self.upConv1(outPut)
        outPut = torch.cat((outPut, copy4)) #* concatenate the tensors
        # print("outPut befor up = ", outPut.shape) #* must be 28x28x1024
        # print("outPut post up = ", outPut.shape) #* must be 28x28x1024

        outPut = self.conv11(outPut)
        outPut = self.conv11_1(outPut)

        outPut = self.upConv2(outPut)
        outPut = torch.cat((outPut, copy3))

        outPut = self.conv12(outPut)
        outPut = self.conv12_1(outPut)

        outPut = self.upConv3(outPut)
        outPut = torch.cat((outPut, copy2))

        outPut = self.conv13(outPut)
        outPut = self.conv13_1(outPut)

        outPut = self.upConv4(outPut)
        outPut = torch.cat((outPut, copy1))

        outPut = self.conv14(outPut)
        outPut = self.conv14_1(outPut)
        outPut = self.lastConv(outPut)

        return outPut




