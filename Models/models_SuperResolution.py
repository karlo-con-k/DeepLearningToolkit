import torch
import torch.nn as nn



class modelSuperResolution(nn.Module):
    '''
        Model img to img, for upgrate the definition of the img.

        Parameters
        ----------
        scale_factor : int
        HEIGHT : int
            Height of the img input
        WIDTH : int
            Width of the img input
        in_CHANNELS : int
            Channels of the img input
    '''

    def __init__(self, scale_factor = 2, HEIGHT = 256, 
                WIDTH = 256, in_CHANNELS = 3):
        super(modelSuperResolution, self).__init__()


        self.scale_factor = scale_factor
        
        self.Sect1  = nn.Sequential(
                nn.Conv2d(
                        in_channels = in_CHANNELS, 
                        out_channels = 64, 
                        kernel_size = 9, 
                        padding = 4),
                nn.SiLU()
        ) 

        self.Sect2  = nn.Sequential(
                nn.Conv2d(
                        in_channels = 64, 
                        out_channels = 32, 
                        kernel_size = 9, 
                        padding = 4),
                nn.SiLU()
        ) 
        self.Sect3  = nn.Sequential(
                nn.Conv2d(
                        in_channels = 32, 
                        out_channels = 3 * (self.scale_factor ** 2), 
                        kernel_size = 5, 
                        padding = 2),
                nn.PixelShuffle(self.scale_factor),
                nn.Sigmoid()
        )


    def forward(self, data):

        # print("data.shape = ", data.shape)
        out = self.Sect1(data)
        # print("out.shape = ", out.shape)
        out = self.Sect2(out)
        # print("out.shape = ", out.shape)
        out = self.Sect3(out)

        return out
    

    def __str__(self):
        print("Summary of model")
        return "model sumarry"
