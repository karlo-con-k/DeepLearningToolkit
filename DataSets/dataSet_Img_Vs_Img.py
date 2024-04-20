import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms




identity_transform = transforms.Compose([
                        #* ToPILImage = cahnge the data type from PyTorch tensor or a NumPy ndarray to : A PIL (Python Imaging Library)
                        transforms.ToPILImage(),
                        #* change the data type from Numpy or PIL to tensor
                        transforms.ToTensor()
                    ])



class DataSet_Img_To_Img(Dataset):
    '''
        A dataset of models from images to labels. In this case, is for a classifier cats vs dogs
        root_Data[0]       = Path to the data images folder for the inPut
        root_Data[1]       = Path to the data images folder for the outPut
        transfor_In_img    = Transformation for the inPut images
        transfor_Out_img   = Transformation for the outPut images
        test            = If is true we will return a dataset of 10 elements for do testing
    ''' 

    def __init__(self, root_Data,
                transfor_In_img   = identity_transform, 
                transfor_Out_img = identity_transform, 
                test     = False, 
                dataSize = 100):
        super(DataSet_Img_To_Img, self).__init__()


        self.data = []
        self.root_Data   = root_Data
        self.transfor_InPut_img  = transfor_In_img
        self.transfor_OutPut_img = transfor_Out_img

        
        #* Create a list of the name of the files in the root_Datas.
        inPut_Images  = os.listdir(self.root_Data[0])
        outPut_Images = os.listdir(self.root_Data[1])


        if(test == True):
            dataSize = min(len(inPut_Images), dataSize)
            inPut_Images  =  inPut_Images[0:dataSize]
            outPut_Images = outPut_Images[0:dataSize]

        #* Save a list of tuples like ([inPut_img.1.jpg, outPut_img1.1.jpg])
        self.data = list(zip(inPut_Images, outPut_Images))

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):

        #* Read the images, transform them into an array, and only use the first 3 channels.
        inPut_Img_file = self.data[index][0]
        # print(self.root_Data[0] + '/' + inPut_Img_file)
        inPut_Img_pth = os.path.join(self.root_Data[0] + '/' + inPut_Img_file)
        inPut_Img  = np.array(Image.open(inPut_Img_pth))
        inPut_Img  = inPut_Img [:, :, :3]
        # print("input = ", inPut_Img)
        
        outPut_Img_file = self.data[index][1]
        # print(self.root_Data[1] + '/' + outPut_Img_file)
        outPut_Img_pth = os.path.join(self.root_Data[1] + '/' + outPut_Img_file)
        outPut_Img  = np.array(Image.open(outPut_Img_pth))
        outPut_Img  = outPut_Img [:, :, :3]


        #* Apply the corresponding trasformations. Could be data aumentation functions
        inPut_Img  = self.transfor_InPut_img(inPut_Img)
        outPut_Img = self.transfor_OutPut_img(outPut_Img)


        return inPut_Img, outPut_Img

