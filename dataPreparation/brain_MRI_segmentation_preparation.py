import os
import random



def copyInNewFile(oldPath : str, newPath : str, imgName : str, deleteOldFile : bool = False):
    '''
        This function copy the img from  direction oldPath/imgName in to newPath/imgName2
        where imgName2 = imgName with out "_mask" string. If deleteOldFile = true the old
        img will be remove. 

        Args:
        -----
            oldPath : str
                The folder where is the img.
            newPath : str
                The folder where we will do the copy of the img.
            imgName : str
                The img name.
            deleteOldFile : bool = False
                deleteOldFile = true if we what to delete the old file
    '''

    with open(oldPath + '/' + imgName, "rb") as f:
        imgCopy = f.read()

    if("mask" in imgName):
        imgName = imgName.replace("_mask", "")

    with open(newPath + '/' + imgName, "wb") as f:
        
        f.write(imgCopy)

    if deleteOldFile == True:
        #TODO test
        os.remove(oldPath)


def split_dataSet(newFoldersPath : str):
    '''
        This function that split the data set in to train and validation folders. 
        The validation set will be the 20% of the origen data set. We need train
        and validation folders in newFoldersPath and img, mask folders in train
        and validation respectivy. And the 100 fo the data set in train. Like this:

        newFoldersPath/
        |
        |--train/
        |    |
        |    |- img      
        |    |
        |    L mask      
        |
        |--validation
        |    |
        |    |- img      
        |    |
        |    L mask

        Args:
        -----
            newFoldersPath : str
                The folder where is the data set with the folders train with img folder and mask folder
                and the folder validation with the folders img and mask (like the diagram).
    '''

    filesImg  = os.listdir(newFoldersPath + "/train/img")
    filesMask = os.listdir(newFoldersPath + "/train/mask")
    dataSetSize = len(filesImg)     #* number of pairs (img, mask) in the data set
    validationSize = dataSetSize//5 #todo add the arg porcentage.

    for _ in range(validationSize):

        imgName = random.choice(filesImg)
        oldPath = newFoldersPath + "/train/img" 
        newPath = newFoldersPath + "/validation/img"
        copyInNewFile(oldPath = oldPath, newPath = newPath, imgName = imgName)
        filesImg.remove(imgName)

        oldPath = newFoldersPath + "/train/mask" 
        newPath = newFoldersPath + "/validation/mask"
        copyInNewFile(oldPath = oldPath, newPath = newPath, imgName = imgName)
        filesMask.remove(imgName)
    
    print("validationSize = ", validationSize)
    print("trainSize = "     , len(filesImg))


def preparation_brain_MRI_Set(pathDataSet : str, newFoldersPath : str):
    '''
        This function prepare the Brain MRI segmentation dataset in the specified
        'newFoldersPath' for training a model image to image. It creates the 
        following folders within 'newFoldersPath': train/img, train/mask, 
        validation/img, and validation/mask. After that the function copies the
        original images into train/img and train/mask. When copying the mask 
        imgaes, "_mask" is removed from their filenames to match with the images 
        filesnames (this is necesary for  use tiwh the DataSet_Img_To_Img class).
        Subsequently, the data is split into 80% for training and 20% for validation.

        Args:
        -----
            pathDataSet : str
                The foder of the data set Brain MRI 
            newFoldersPath : str
                    The folder where we will create the folders train/img, train/mask \n
                    validation/img, and validation/mask. For afther move the data set \n
                    in the train folder, and after split the data set in train, and
                    validation folders.
    '''

    folder_kaggle_3m = os.listdir(pathDataSet)
    pathDataSet += '/' + folder_kaggle_3m[0]
    foldersList  = os.listdir(pathDataSet)[2:] #* ignore the red and csv

    os.makedirs(newFoldersPath + "/train/img" , exist_ok=True) #*Create the folders for save the imgs
    os.makedirs(newFoldersPath + "/train/mask", exist_ok=True) #*Create the folders for save the imgs
    os.makedirs(newFoldersPath + "/validation/img" , exist_ok=True) #*Create the folders for save the imgs
    os.makedirs(newFoldersPath + "/validation/mask", exist_ok=True) #*Create the folders for save the imgs

    dataSetSize = 0
    #* Open all the img in the dataSet and separate in img and mask. 
    for folder in (foldersList):
        imgFolderList = os.listdir(pathDataSet + '/' + folder)
        for  imgName in (imgFolderList):
            dataSetSize += 1
            if("mask" in imgName):
                copyInNewFile(pathDataSet + '/' +  folder, newFoldersPath + "/train/mask", imgName)
            else:
                copyInNewFile(pathDataSet + '/' +  folder, newFoldersPath + "/train/img", imgName)

    split_dataSet(newFoldersPath)


