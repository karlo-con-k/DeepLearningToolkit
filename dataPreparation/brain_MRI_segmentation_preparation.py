import os



def copyInNewFile(oldPath :str, newPath : str, imgName : str):
    print("oldPath = ", oldPath)
    print("newPath = ", newPath)

    if("mask" in imgName):
        imgName = imgName.replace("_mask", "")

    with open(oldPath, "rb") as f:
        imgCopy = f.read()
    
    with open(newPath + '/' + imgName, "wb") as f:
        
        f.write(imgCopy)




def preparation_brain_MRI_Set(pathDataSet : str, newFoldersPath : str):
    '''
        Function for prepar the data set Brain MRI segmentation, for after use the class Dataset img to img. We need two folders with the img, and the mask.
    '''

    folder_kaggle_3m = os.listdir(pathDataSet)
    pathDataSet += '/' + folder_kaggle_3m[0]
    foldersList  = os.listdir(pathDataSet)[2:] #* ignore the red and csv

    os.makedirs(newFoldersPath + "/img" , exist_ok=True) #*Create the folders for save the imgs
    os.makedirs(newFoldersPath + "/mask", exist_ok=True) #*Create the folders for save the imgs

    dataSetSize = 0
    #* Open all the img in the dataSet and separate in img and mask. 
    for folder in (foldersList):
        imgFolderList = os.listdir(pathDataSet + '/' + folder)
        for  imgName in (imgFolderList):
            dataSetSize += 1
            if("mask" in imgName):
                # imgName = imgName.replace("_mask", "")
                copyInNewFile(pathDataSet + '/' +  folder + '/' + imgName, newFoldersPath + "/mask", imgName)
            else:
                copyInNewFile(pathDataSet + '/' +  folder + '/' + imgName, newFoldersPath + "/img", imgName)

    print("dataSetSize = ", dataSetSize)
    dataSetSize *= 0.75
    dataSetSize = int(dataSetSize)
    print("dataSetSize * 0.75 = ", dataSetSize)









