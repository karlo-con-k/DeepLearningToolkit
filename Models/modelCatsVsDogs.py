import torch
import torch.nn as nn

class modelCNN(nn.Module):
    '''
        Create a model such that geave a image and return a value.
    '''


    # todo add historial in models.

    def __init__(self):
        super(modelCNN, self).__init__()
        '''
            
        '''

        self.conv_1 = nn.Conv2d(in_channels = 3 , out_channels = 32 , kernel_size = 3) #! limit
        self.pool_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_2 = nn.Conv2d(in_channels = 32, out_channels = 64 , kernel_size = 3) 
        self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.pool_3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.flat_1 = nn.Flatten()
        self.dens_1 = nn.Sequential(
                                    nn.Linear(in_features = 107648, out_features = 128),
                                    nn.ReLU(),
                                    nn.Linear(in_features = 128,    out_features = 64),
                                    nn.ReLU(),
                                    nn.Linear(in_features = 64, out_features = 1)
                                    # nn.Sigmoid()
                                    )

    def forward(self, data):

        out = self.conv_1(data)
        out = self.pool_1(out)
        out = self.conv_2(out)
        out = self.pool_2(out)
        out = self.conv_3(out)
        out = self.pool_3(out)
        out = self.flat_1(out)
        out = self.dens_1(out)
        return out;


    def __str__(self):
        #todo print the summary of the model
        return "model.summary"


