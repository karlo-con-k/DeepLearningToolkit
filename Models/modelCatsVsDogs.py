import torch
import torch.nn as nn

class modelCNN(nn.Module):
    '''
        Create a model such that geave a image and return a value.
    '''


    def __init__(self):
        super(modelCNN, self).__init__()
        '''
            
        '''

        self.Conv_1 = nn.Conv2d(in_channels = 3 , out_channels = 16 , kernel_size = 3, padding = 0, stride = 2) #! limit
        self.Norm_1 = nn.BatchNorm2d(16)
        self.Relu_1 = nn.ReLU()
        self.Pool_1 = nn.MaxPool2d(2)

        self.Conv_2 = nn.Conv2d(in_channels = 16, out_channels = 32 , kernel_size = 3) 
        self.Norm_2 = nn.BatchNorm2d(32)
        self.Relu_2 = nn.ReLU()
        self.Pool_2 = nn.MaxPool2d(2)
        
        self.Conv_3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 0, stride = 2)
        self.Norm_3 = nn.BatchNorm2d(64)
        self.Relu_3 = nn.ReLU()
        self.Pool_3 = nn.MaxPool2d(2)
        
        self.flat_1 = nn.Linear(3136, 10)  
        self.dens_1 = nn.Sequential(
                                    nn.Linear(in_features = 10, out_features = 2),
                                    # nn.ReLU(),
                                    # nn.ReLU()
                                    )
        self.Relu = nn.ReLU()


    def forward(self, data):

        out = self.Conv_1(data)
        out = self.Norm_1(out)
        out = self.Relu_1(out)
        out = self.Pool_1(out)

        out = self.Conv_2(out)
        out = self.Norm_2(out)
        out = self.Relu_2(out)
        out = self.Pool_2(out)

        out = self.Conv_3(out)
        out = self.Norm_3(out)
        out = self.Relu_3(out)
        out = self.Pool_3(out)

        out = out.view(out.size(0),-1)
        
        out = self.Relu(self.flat_1(out))
        out = self.dens_1(out)
        # out = self.relu_1(out)
        return out


    def __str__(self):
        #todo print the summary of the model
        return "model.summary"


class modelCNN2(nn.Module):
    '''
        Create a model such that geave a image and return a value.
    '''


    def __init__(self):
        super(modelCNN2, self).__init__()
        '''
            
        '''

        self.Sect_1 = nn.Sequential(
            nn.Conv2d(in_channels = 3 , out_channels = 16 , kernel_size = 3, padding = 0, stride = 2), #! limit
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.Sect_2 = nn.Sequential(
            nn.Conv2d(in_channels = 16 , out_channels = 32 , kernel_size = 3, padding = 0, stride = 2), #! limit
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.Sect_3 = nn.Sequential(
            nn.Conv2d(in_channels = 32 , out_channels = 64 , kernel_size = 3, padding = 0, stride = 2), #! limit
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flat_1 = nn.Linear(3*3*64, 10)  
        self.dens_1 = nn.Sequential(
                                    nn.Linear(in_features = 10, out_features = 2),
                                    # nn.ReLU(),
                                    # nn.ReLU()
                                    )
        self.Relu = nn.ReLU()


    def forward(self, data):

        out = self.Sect_1(data)
        out = self.Sect_2(out)
        out = self.Sect_3(out)

        out = out.view(out.size(0),-1)
        
        out = self.Relu(self.flat_1(out))
        out = self.dens_1(out)
        # out = self.relu_1(out)
        return out


    def __str__(self):
        #todo print the summary of the model
        return "model.summary"

