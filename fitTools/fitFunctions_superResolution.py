
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader



def train_modelSuperResolution(data_loader, model, opt_model, device, 
                    data_loader_Val = None, 
                    num_epochs = 1, criterion = F.mse_loss, 
                    get_History = True, getValLoos = False,
                    model_save_dir = None):
    '''
        Fit function for a clasificator cat vs dog model. This function will 
        save the models in each epoch, and it return the history if we whant.

        data_loader     = index list of the batch tensors of the dataset
        model           = model to fit
        opt_model       = optimization algorihm
        device          = device that will use like GPU
        data_loader_Val = index list of the batch tensors of the dataset validation
        get_History     = if we whant to get the historial of the model 
        history         = if true, then create a history dictionary and will append the values loss, and accuracy
        criterion       = losst function of the model
        getValLoos      = if we whant to get the loos in the validation dataset during the training  
    '''

    if(get_History):
        history = {
                    "MAE" : [],
                    "ACC" : [] #todo add the ACC
                    }
        if(getValLoos == True):
            history['MAE_Val'] = []

    model.to(device)
    model.train()
    sizeDataSet = len(data_loader.dataset)          #* size of the dataSet i.e number of images
    
    if(data_loader_Val is not None):
        sizeDataSetVal = len(data_loader_Val.dataset)      #* size of the dataSetVal 
    else:
        sizeDataSetVal = 1

    batch_size  = data_loader.batch_size


    for epoch in range(num_epochs):

        #* This will be the Mean Absolute Error of the dataSet 
        model_total_loss  = 0
        loop = tqdm(enumerate(data_loader), total = len(data_loader))
        for batch_idx, (imgInput, imgOutput) in loop:
            imgInput  = imgInput.to(device)
            imgOutput = imgOutput.to(device) #! imgOutput = imgOutput.float().to(device) 
            opt_model.zero_grad()
            prediction = model(imgInput)
            loss  = criterion(prediction, imgOutput)
            #* loss.item() for get the value of the tensor.
            model_total_loss += loss.item()*batch_size      
            loss.backward()
            opt_model.step()
            loop.set_description(f"Epoch {epoch+1}/{num_epochs} process: {int((batch_idx / len(data_loader)) * 100)}")
            loop.set_postfix(modelLoss = loss.data.item())

        avg_loss = model_total_loss / sizeDataSet
        print(f'Epoch completed, Average Loss: {avg_loss:.4f}')

        if(get_History): 
            history["MAE"].append(model_total_loss/sizeDataSet) 
            #TODO define well the ACC
            history["ACC"].append(model_total_loss/sizeDataSet) 


        if(getValLoos == True): 

            val_MSE = getMAE(model = model, 
                            data_loader = data_loader,
                            criterion = criterion,
                            batch_size = batch_size,
                            device = device)

            if(get_History):
                history["MAE_Val"].append(val_MSE/sizeDataSetVal)
            print(f'Epoch completed, Average Val Loss: {(val_MSE/sizeDataSetVal):.4f}')

        #* Save the model afther the epoch
        torch.save({ 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': opt_model.state_dict(), 
            }, os.path.join(model_save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

    if(get_History):
        return history
    

def getMAE(model, data_loader, criterion, batch_size, device):
    '''
        Parameters
        -----------
        data_loader : DataLoader  
            Index list of the batch tensors of the dataset
        model : nn.Module
            Model to fit
        criterion :  torch.nn.Module, optional
            Loss function of the model
        batch_size : int
            Batch size
        device : str
            Device that we use like GPU

        Returns
        -------
        Return the accuracy of the model in the data loader
    '''

    #Todo test this function   
    #* Set the model to evaluation mode
    model.eval()

    with torch.no_grad(): #* Disable gradient computation
        model_total_loss = 0

        for idx, imgsInput, imgsOutput in enumerate(data_loader):
            imgsInput  = imgsInput.to(device)
            imgsOutput = imgsOutput.to(device)
            prediction = model(imgsInput)
            loss       = criterion(imgsInput, prediction)

            model_total_loss += loss.item()*batch_size  

    return model_total_loss



class fitertImgToImg():
    '''
        Base class for img to img models
    '''

    def __init__(self,
                model : nn.Module, 
                dataSet, 
                device     : str = "cpu",
                batch_size : int = 64,
                dataSet_Val = None, 
                model_save_dir : str =  None):

        if(model_save_dir is None):
            raise ValueError('model_save_dir could not be None')

        #* Initialize history
        self.history = {
            "train_MAE" : [],
            "val_MAE"   : [],
            #todo add the PSNR, or SSIM, FID
        }

        #* Start the class attributes
        self.dataSet     = dataSet
        self.model       = model
        self.device      = device
        self.batch_size  = batch_size
        self.dataSet_Val = dataSet_Val
        self.model_save_dir  = model_save_dir

        #* Initialize data loaders
        self.data_loader = DataLoader(
                                self.dataSet, 
                                batch_size  = self.batch_size,
                                num_workers = 0,
                                shuffle = True)
        
        if(self.dataSet_Val is not None):
            self.data_loader_Val = DataLoader(
                                        self.dataSet_Val,
                                        batch_size  = self.batch_size,
                                        num_workers = 0,
                                        shuffle = True
            )

    def getMAE(self,
        data_loader : DataLoader, 
        criterion : torch.nn.Module = torch.nn.CrossEntropyLoss(),
        ):

        size_Data_loader = len(data_loader.dataset)
        model_MAE = 0

        if size_Data_loader == 0:
            raise ValueError('The data set should not be empty.')

        with torch.no_grad():
            for (imgInput, imgOutPut) in data_loader:
                imgInput    =  imgInput.to(self.device)
                imgOutPut   = imgOutPut.to(self.device)
                modelOutPut = self.model(imgInput)

                loss = criterion(imgOutPut, modelOutPut)
                model_MAE += loss.item()*self.batch_size

        return model_MAE/size_Data_loader

    def trainModel(self,
                opt_model  : torch.optim, 
                criterion  : torch.nn.Module = torch.nn.CrossEntropyLoss(),
                num_epochs : int = 1):
        '''
            Funtion for train the model
        '''

        if num_epochs < 0:
            raise ValueError('num_epochs should be non-negative')
        
        sizeDataSet =len(self.data_loader.dataset)
        self.model.to(self.device)

        for epoch in range(num_epochs):
            loop = tqdm(enumerate(self.data_loader), total = len(self.data_loader))
            self.model.train() #* model in train mood
            train_MAE = 0
            for batch_idx, (imgInput, imgOutPut) in loop:
                imgInput  =  imgInput.to(self.device)
                imgOutPut = imgOutPut.to(self.device)
                opt_model.zero_grad()
                modelOutPut = self.model(imgInput)
                
                #* Get the batch loss and computing train MAE
                loss       = criterion(modelOutPut, imgOutPut)
                train_MAE += loss.item()*self.batch_size

                #* Get gradients, and update the parameters of the model
                loss.backward()
                opt_model.step()

                #* Plot the loss and the progress bar
                loop.set_description(f"Epoch {epoch+1}/{num_epochs} process: {int((batch_idx / len(self.data_loader)) * 100)}")
                loop.set_postfix(modelLoss = loss.data.item())

            train_MAE = train_MAE / sizeDataSet
            print(f'Epoch completed, TRAIN MAE {train_MAE:.4f}')
            self.history["train_MAE"].append(train_MAE)

            if(self.dataSet_Val is not None):
                val_MAE = self.getMAE(data_loader = self.data_loader_Val, criterion = criterion)
                #* We could try a diferent criterio for the val case in the same dataset.

                print(f'Epoch completed, VAL MAE: {(val_MAE):4f}')
                self.history["val_MAE"].append(val_MAE)

                #* Save the best model in val_MAE
                if(val_MAE >= min(self.history["val_MAE"])):
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict':opt_model.state_dict()
                    }, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch + 1}_Val_MAE_{"{:.3f}".format(val_MAE)}.pt'))
            
            #* If we don have val_MAE, save in function of train_MAE
            else:
                if(train_MAE >= min(self.history["train_MAE"])):
                    torch.save({
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer_state_dict' : opt_model.state_dict()
                    }, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch + 1}_Train_MAE_{"{:.3f}".format(train_MAE)}.pt'))





class fiterU_Net(fitertImgToImg):

    def __init__(self, 
                model: nn.Module, 
                dataSet, 
                device: str = "cpu", 
                batch_size: int = 64, 
                dataSet_Val = None, 
                model_save_dir: str = None):
        super().__init__(model, dataSet, device, batch_size, dataSet_Val, model_save_dir)

    def trainModel(self,
                opt_model  : torch.optim, 
                criterion  : torch.nn.Module = torch.nn.CrossEntropyLoss(),
                num_epochs : int = 1):
        '''
            Funtion for train the model U-Net, we only change the line 
            loss = criterion(modelOutPut, imgOutPut) for the line
            loss = criterion(modelOutPut[:,0,:,:],imgOutPut[:,0,:,:])
        '''

        if num_epochs < 0:
            raise ValueError('num_epochs should be non-negative')
        
        sizeDataSet =len(self.data_loader.dataset)
        self.model.to(self.device)

        for epoch in range(num_epochs):
            loop = tqdm(enumerate(self.data_loader), total = len(self.data_loader))
            self.model.train() #* model in train mood
            train_MAE = 0
            for batch_idx, (imgInput, imgOutPut) in loop:
                imgInput  =  imgInput.to(self.device)
                imgOutPut = imgOutPut.to(self.device)
                opt_model.zero_grad()
                modelOutPut = self.model(imgInput)
                
                #* Get the batch loss and computing train MAE
                loss       = criterion(modelOutPut[:,0,:,:],imgOutPut[:,0,:,:])
                train_MAE += loss.item()*self.batch_size

                #* Get gradients, and update the parameters of the model
                loss.backward()
                opt_model.step()

                #* Plot the loss and the progress bar
                loop.set_description(f"Epoch {epoch+1}/{num_epochs} process: {int((batch_idx / len(self.data_loader)) * 100)}")
                loop.set_postfix(modelLoss = loss.data.item())

            train_MAE = train_MAE / sizeDataSet
            print(f'Epoch completed, TRAIN MAE {train_MAE:.4f}')
            self.history["train_MAE"].append(train_MAE)

            if(self.dataSet_Val is not None):
                val_MAE = self.getMAE(data_loader = self.data_loader_Val, criterion = criterion)
                #* We could try a diferent criterio for the val case in the same dataset.

                print(f'Epoch completed, VAL MAE: {(val_MAE):4f}')
                self.history["val_MAE"].append(val_MAE)

                #* Save the best model in val_MAE
                if(val_MAE >= min(self.history["val_MAE"])):
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict':opt_model.state_dict()
                    }, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch + 1}_Val_MAE_{"{:.3f}".format(val_MAE)}.pt'))
            
            #* If we don have val_MAE, save in function of train_MAE
            else:
                if(train_MAE >= min(self.history["train_MAE"])):
                    torch.save({
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer_state_dict' : opt_model.state_dict()
                    }, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch + 1}_Train_MAE_{"{:.3f}".format(train_MAE)}.pt'))




