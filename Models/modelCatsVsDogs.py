import torch
import torch.nn as nn


class modelCNN2(nn.Module):
    '''
        Create a model such that geave a image and return two values (p1,p2), 
        where p1 is the probability that the image is of the classe 1 and similary for p2.
    '''


    def __init__(self):
        super(modelCNN2, self).__init__()
        # todo define usin HEIGHT, WIDTH AND CHENELS OF THE IMAGE

        self.Sect_1 = nn.Sequential(
                nn.Conv2d(
                        in_channels  = 3, 
                        out_channels = 16, 
                        kernel_size  = 3,
                        padding      = 0, 
                        stride       = 2), #! limit
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
        )

        self.Sect_2 = nn.Sequential(
                nn.Conv2d(
                        in_channels  = 16, 
                        out_channels = 32, 
                        kernel_size  = 3, 
                        padding      = 0, 
                        stride       = 2), #! limit
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
        )
        
        self.Sect_3 = nn.Sequential(
                nn.Conv2d(
                        in_channels  = 32, 
                        out_channels = 64, 
                        kernel_size  = 3, 
                        padding      = 0, 
                        stride       = 2), #! limit
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
        )

        self.flat_1 = nn.Linear(3*3*64, 10)  
        self.Relu   = nn.ReLU()

        self.dens_1 = nn.Sequential(
            nn.Linear(in_features = 10, out_features = 2),
        )


    def forward(self, data):
        
        out = self.Sect_1(data)
        out = self.Sect_2(out)
        out = self.Sect_3(out)

        #* change the dimentions of the tensor of using the flat_1             
        out = out.view(out.size(0),-1)

        out = self.flat_1(out)
        out = self.Relu(out)
        out = self.dens_1(out)

        return out


    def __str__(self):
        #todo print the summary of the model
        return "model.summary"


