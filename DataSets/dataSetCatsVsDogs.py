
import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

import torchvision.transforms as transforms

identity_transform = transforms.Compose([
                        #* Compose = compoues a list of function
                        transforms.ToPILImage(),
                        #* ToPILImage = cahnge the data type from PyTorch tensor or a NumPy ndarray to : A PIL (Python Imaging Library)
                        transforms.ToTensor(), 
                        #* change the data type from Numpy or PIL to tensor
                        ])


class DataSet_Img_To_Label(Dataset):
    def __init__(self, root_Data,
                transform_img = identity_transform, transform_label = None, test = False):
        super(DataSet_Img_To_Label, self).__init__()

        '''
            A dataset of models from images to labels. In this case, is for a classifier cats vs dogs

            root_Data       = Path to the data images folder
            transform_img   = Transformation for the images
            transform_label = Transformation for the label
            test            = If is true we will return a dataset of 10 elements for do testing
        ''' 

        self.data = []
        self.root_Data       = root_Data
        self.transform_img   = transform_img
        self.transform_label = transform_label

        #* Create a list of the name of the files in the root_Data.
        images = os.listdir(self.root_Data)

        if(test == True): #* return a test DataSet
            images = images[0:10]

        print("images.size() = ", len(images))

        list_Label = ["0"]*len(images) 

        #* cats = 1, dogs = 0
        for idx, name_image in enumerate(images):
            if(name_image[0] == 'd'):
                list_Label[idx] = 0

            elif(name_image[0] == 'c'):
                list_Label[idx] = 1

        #* Save a list of tuples with the file name and the label like this [('cat.0.jpg', 1)]
        self.data = list(zip(images, list_Label)) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index = None):
        '''
            If index != None return the image of the position index and the label. 
            In other case index = random and do the same.

            Index = Image position that we will return.
        '''

        if(index != None):
            img_file, label_file = self.data[index]
        else:
            img_file = random.choice(self.images)

        #* Read the image, transform it into an array, and only use the first 3 channels.
        img_pth = os.path.join(self.root_Data + '/' + img_file)
        img  = np.array(Image.open(img_pth))
        img  = img [:, :, :3]


        #* Apply the corresponding transformation to the data.
        if(self.transform_img != None):
            img = self.transform_img(img)

        if(self.transform_label != None):
            mask = self.transform_label(mask)

        return img, label_file

    def __str__(self):
        
        print("len(dataSet) = ",len(self.data))
        if(len(self.data) > 0):
            print("dataSet[0]   = ",self.data[0])

        return ""
