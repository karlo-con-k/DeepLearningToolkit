
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from TestNotebooks.toolsTest import plot_img_tensor


class model_U_Net(nn.Module):

    '''
        Implementation of the model U-Net for img segmentation

        Attributes
        ----------
            downsampling1 : nn.Sequential
                A sequential aplication of conv layers.
            downsampling2 : nn.Sequential
                A sequential aplication of MaxPool2d layers, and Conv layers.
            downsampling3 : nn.Sequential
                A sequential aplication of MaxPool2d layers, and Conv layers.
            downsampling4 : nn.Sequential
                A sequential aplication of MaxPool2d layers, and Conv layers.
            centerBlock : nn.Sequential
                A sequential aplication of MaxPool2d, Conv, and ConvTranspose2d layers.
            upsampling1 : nn.Sequential
                A sequential aplication of Conv, and ConvTranspose2d
            upsampling2 : nn.Sequential
                A sequential aplication of Conv, and ConvTranspose2d
            upsampling3 : nn.Sequential
                A sequential aplication of Conv, and ConvTranspose2d
            upsampling4 : nn.Sequential
                A sequential aplication of Conv, and ConvTranspose2d
            softMax : nn.Softmax
                A softmax activation function for get probabilitys in the end of the model
    
        Methods
        -------
            onv_block( 
                        in_channels  : int,
                        out_channels : int, 
                        kernel_size  : int, 
                        stride : int
                    ) -> nn.Sequential:
                This method will return an sequential with Conv2d, SiLU, BatchNorm2d, 
                Conv2d, and SilU.

            convUp_block(, 
                in_channels  : int,
                out_channels : int, 
                kernel_size  : int, 
                stride : int
                ) -> nn.Sequential:
                This method will return an sequential with ConvTransposed2d, BatchNorm2d,
                and SiLU.
            forward(inPut)
                This function return the model prediction
    '''

    def __init__(self, 
                HEIGHT : int = 572,
                WIDTH  : int = 572,
                in_CHANNELS: int = 1):
        super(model_U_Net, self).__init__()

        #* -> -> (U-net architecture arrows)
        self.downsampling1 = self.conv_block(in_CHANNELS , 64, 3, 1)  

        #* ↓ -> ->
        self.downsampling2 = nn.Sequential(
                                            nn.MaxPool2d(kernel_size = 2),
                                            self.conv_block(64 , 128, 3, 1)
                                        )

        #* ↓ -> ->
        self.downsampling3 = nn.Sequential(
                                            nn.MaxPool2d(kernel_size = 2),
                                            self.conv_block(128, 256, 3, 1)
                                        )

        #* ↓ -> ->
        self.downsampling4 = nn.Sequential(
                                            nn.MaxPool2d(kernel_size = 2),
                                            self.conv_block(256, 512, 3, 1)
                                        )

        #* ↓ -> ->  ↑
        #* use pixshuffle ? 
        self.centerBlock = nn.Sequential(
                                            nn.MaxPool2d(kernel_size = 2),
                                            self.conv_block(512 , 1024, 3, 1),
                                            self.convUp_block(1024, 512, 3, 2)
                                        )

        #* ↑ -> -> 
        self.upsampling1 = nn.Sequential(
                                            self.conv_block(1024, 512, 3, 1),
                                            self.convUp_block(512, 256, 3, 2)
                                        )

        #* ↑ -> -> 
        self.upsampling2 = nn.Sequential(
                                            self.conv_block(512, 256, 3, 1),
                                            self.convUp_block(256, 128, 3, 2)
                                        )

        #* ↑ -> -> 
        self.upsampling3 = nn.Sequential(
                                            self.conv_block(256, 128, 3, 1),
                                            self.convUp_block(128, 64, 3 ,2)
                                        )

        #* ↑ -> -> 
        self.upsampling4 = nn.Sequential(
                                            self.conv_block(128, 64, 3, 1), 
                                            self.conv_block(64, 2, 1, 1)
                                        )

        self.softMax  = nn.Softmax(dim = 1)

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
            nn.SiLU(), #* the origin model use Relu
            nn.BatchNorm2d(out_channels), #* The origin model do not use batchNorm
            nn.Conv2d(
                    in_channels  = out_channels,
                    out_channels = out_channels,
                    kernel_size  = kernel_size,
                    stride = stride
                    ), 
            nn.SiLU() #* the origin model use Relu
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
            nn.BatchNorm2d(out_channels), 
            nn.SiLU()
        )

    def forward(self, inPut):

        outPut = self.downsampling1(inPut)                  #* -> -> 
        copy1  = transforms.Resize((392, 392))(outPut)      

        outPut = self.downsampling2(outPut)                 #* ↓ -> ->
        copy2  = outPut[:, :, 40:240, 40:240]

        outPut = self.downsampling3(outPut)                 #* ↓ -> ->
        copy3  = outPut[:, :, 16:120, 16:120] 

        outPut = self.downsampling4(outPut)                 #* ↓ -> ->
        copy4  = outPut[:, :, 4:60, 4:60] #todo maxPol ? 

        outPut = self.centerBlock(outPut)                   #* ↓ -> ->  ↑

        outPut = torch.cat((outPut, copy4), dim=1)          #* concatenate the tensors
        outPut = self.upsampling1(outPut)                   #* ↑ -> ->  

        outPut = torch.cat((outPut, copy3), dim = 1) 
        outPut = self.upsampling2(outPut)                   #* ↑ -> ->  

        outPut = torch.cat((outPut, copy2), dim= 1)
        outPut = self.upsampling3(outPut)                   #* ↑ -> ->  

        outPut = torch.cat((outPut, copy1), dim= 1)
        outPut = self.upsampling4(outPut)                   #* ↑ -> ->  

        outPut = self.softMax(outPut)
        return outPut


#TODO make a U-Net for every inPut size
from TestNotebooks.toolsTest import plot_img_tensor

class model_u_Net(model_U_Net):
    '''
        Implementation of the model U-Net, but with other inPut size. For have 
        little U-Net model.

        Methods 
            outPutsCopys(inPut)
                This function will return the model prediction, and the outPut of the
                downsampling's layers.
    '''

    def __init__(self, 
                HEIGHT: int = 252, 
                WIDTH : int = 252,
                in_CHANNELS: int = 1):
        super().__init__(HEIGHT, WIDTH, in_CHANNELS)

    def outPutsCopys(self, inPut):
        '''
            This function will compute and return the model output, and the outputs of 
            the fist four blocks. 

            Args
            ----
                inPut : torch.Tensor
                    A img batch tensor of shape (batch size, self.in_CHANNELS, 252, 252).

            Returns
            -------
            A tuple with (outPut, copy1, copy2, copy3, copy4), where outPut is the model 
            output, copy1 is the output of the fist convBlock, ... , copy4 is the output 
            of the fourth convBlock.

        '''

        outPut = self.downsampling1(inPut)                   #* -> -> 
        copy1  = transforms.Resize((72, 72))(outPut)

        outPut = self.downsampling2(outPut)                  #* ↓ -> ->
        copy2  = transforms.Resize((40, 40))(outPut)

        outPut = self.downsampling3(outPut)                  #* ↓ -> ->
        copy3  = transforms.Resize((24, 24))(outPut)  

        outPut = self.downsampling4(outPut)                  #* ↓ -> ->
        copy4  = transforms.Resize((16, 16))(outPut)  

        outPut = self.centerBlock(outPut)                    #* ↓ -> ->  ↑

        outPut = torch.cat((outPut, copy4), dim=1)           #* concatenate the tensors
        outPut = self.upsampling1(outPut)                    #* ↑ -> ->

        outPut = torch.cat((outPut, copy3), dim = 1)  
        outPut = self.upsampling2(outPut)                    #* ↑ -> ->

        outPut = torch.cat((outPut, copy2), dim= 1)
        outPut = self.upsampling3(outPut)                    #* ↑ -> ->

        outPut = torch.cat((outPut, copy1), dim= 1)          
        outPut = self.upsampling4(outPut)                    #* ↑ -> ->

        outPut = self.softMax(outPut)                        #* get probabilitys
        return outPut, copy1, copy2, copy3, copy4

    def forward(self, inPut):

        outPut = self.downsampling1(inPut)                   #* -> -> 
        copy1  = transforms.Resize((72, 72))(outPut)

        outPut = self.downsampling2(outPut)                  #* ↓ -> ->
        copy2  = transforms.Resize((40, 40))(outPut)

        outPut = self.downsampling3(outPut)                  #* ↓ -> ->
        copy3  = transforms.Resize((24, 24))(outPut)  

        outPut = self.downsampling4(outPut)                  #* ↓ -> ->
        copy4  = transforms.Resize((16, 16))(outPut)  

        outPut = self.centerBlock(outPut)                    #* ↓ -> ->  ↑

        outPut = torch.cat((outPut, copy4), dim=1)           #* concatenate the tensors
        outPut = self.upsampling1(outPut)                    #* ↑ -> ->

        outPut = torch.cat((outPut, copy3), dim = 1)  
        outPut = self.upsampling2(outPut)                    #* ↑ -> ->

        outPut = torch.cat((outPut, copy2), dim= 1)
        outPut = self.upsampling3(outPut)                    #* ↑ -> ->

        outPut = torch.cat((outPut, copy1), dim= 1)          
        outPut = self.upsampling4(outPut)                    #* ↑ -> ->

        outPut = self.softMax(outPut)                        #* get probabilitys
        return outPut