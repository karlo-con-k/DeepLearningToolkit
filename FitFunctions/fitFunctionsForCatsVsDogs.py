
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader




def train_modelCNN(data_loader, model, opt_model, device, 
                    data_loader_Val = None, 
                    num_epochs = 1, criterion = torch.nn.CrossEntropyLoss(), 
                    get_History = True, getVal = False,
                    model_save_dir = None):
    '''
        Fit function for a clasificator cat vs dog model. This function will 
        save the models in each epoch, and it return the history if we whant.

        Attributes
        -----------
        data_loader : DataLoader  
            index list of the batch tensors of the dataset
        model : nn.Module
            model to fit
        opt_model : optim  
            optimization algorihm
        device : str
            device that will use like GPU
        data_loader_Val : DataLoader, optional
            index list of the batch tensors of the dataset validation
        num_epochs : int, optional  
            number of epochs to train the model (default is 1)
        criterion :  torch.nn.Module, optional
            loss function of the model
        get_History : bool, optional
            if we whant to get the losst historial of the model 
        getVal : bool, optional 
            If we whant to get the ACC and MAE in validation data loader
        model_save_dir : str, optional
            directory to save the trained model (default is None)

        Returns
        -------
            history of the training process, if get_History is True, otherwise None
    '''
    

    model.to(device)
    
    sizeDataSet = len(data_loader.dataset) 
    batch_size  = data_loader.batch_size
    if(get_History or getVal):
        history = {
                    "train_MAE" : [],
                    "train_ACC" : [] #todo add the ACC
                    }
        if(getVal == True):
            history['val_MAE'] = []
            history['val_ACC'] = []


    for epoch in range(num_epochs):
        model.train()
        train_MAE = 0 #* This will be the MAE of the train dataSet
        train_ACC = 0  #* This will be the ACC of the train dataSet
        loop = tqdm(enumerate(data_loader), total = len(data_loader))
        
        for batch_idx, (imgs, labels) in loop:
            imgs        = imgs.to(device)
            labels      = labels.to(device) #! labels = labels.float().to(device) 
            opt_model.zero_grad()
            outputs = model(imgs)

            #* Get the preditions and the loss for the batch
            loss              = criterion(outputs, labels)  #* get the value of the tensor.
            train_MAE += loss.item()*batch_size #Todo flotant train_ACC problems ??? 
            prediction = torch.max(outputs, 1)[1]  
            train_ACC  += (prediction == labels).sum().item() 

            loss.backward()
            opt_model.step()
            loop.set_description(f"Epoch {epoch+1}/{num_epochs} process: {int((batch_idx / len(data_loader)) * 100)}")
            loop.set_postfix(modelLoss = loss.data.item())

        train_ACC = train_ACC / sizeDataSet
        train_MAE = train_MAE / sizeDataSet
        print(f'Epoch completed, TRAIN MAE: {train_MAE:.4f}')
        print(f'Epoch completed, TRAIN ACC: {train_ACC:.4f}')

        if(get_History): 
            history["train_MAE"].append(train_MAE) 
            history["train_ACC"].append(train_ACC) 

        if(getVal == True):
            ACC_Val, MAE_Val = getAccuracy_and_MAE( model = model, 
                                                    data_loader = data_loader_Val, 
                                                    criterion   = criterion, 
                                                    batch_size  = batch_size, 
                                                    device      = device)

            print(f'Epoch completed, VAL MAE: {(MAE_Val):.4f}')
            print(f'Epoch completed, VAL ACC: {(ACC_Val):.4f}')
            history["val_MAE"].append(MAE_Val)
            history["val_ACC"].append(ACC_Val)

        #todo save the best model
        #* Save the model afther the epoch
        print("model_save_dir =", model_save_dir)
        torch.save({ 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': opt_model.state_dict(), 
            }, os.path.join(model_save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

    if(get_History or getVal):
        return history


def getAccuracy_and_MAE(model, data_loader, criterion, batch_size, device):
    '''
        Parameters
        -----------
        data_loader : DataLoader  
            index list of the batch tensors of the dataset
        model : nn.Module
            model to fit
        device : str
            Device that we use like GPU
        criterion :  torch.nn.Module, optional
            loss function of the model
        batch_size : int
            Size of the batchs in the data loader

        Returns
        -------
        Return a tuple with the accuracy of the model in the data loader and the
        model total loss in the data loader
    '''

    #* Set the model to evaluation mode
    model.eval()
    model_ACC = 0
    model_MAE = 0
    size_data_loader = len(data_loader.dataset)

    #* Disable gradient computation
    with torch.no_grad(): 
        for imgs, labels in data_loader:
            labels  = labels.to(device)
            imgs    = imgs.to(device)
            outputs = model(imgs)

            batch_loss = criterion(outputs, labels)  
            model_MAE += batch_loss.item()*batch_size

            #* Get the index of the maximum predicted value
            prediction = torch.max(outputs, 1)[1]  
            #* Count the number of correct predictions
            model_ACC += (prediction == labels).sum().item()  

    model_ACC = model_ACC / size_data_loader  # Compute the accuracy
    model_MAE = model_MAE / size_data_loader

    return model_ACC, model_MAE


class trainerCatsVsDogs:
    '''
        This class trains a classifier model for distinguishing between cats and dogs. 
        It saves the models at the end of each epoch and returns the training history.

        Attributes
        -----------
        model : nn.Module
            The model to be trained.
        dataSet : Dataset
            The dataset containing the training data.
        device : str
            The device (e.g., 'cuda' or 'cpu') where the training will take place.
        batch_size : int, optional
            The batch size used for training (default is 64).
        dataSet_Val : Dataset, optional
            The dataset containing the validation data (default is None).
        model_save_dir : str, optional
            The directory where trained models will be saved (default is None).

        Methods
        -------
        fitModel(opt_model  : torch.optim, 
                criterion  : torch.nn.Module = torch.nn.CrossEntropyLoss(),
                num_epochs : int = 1)
            Train the model for the specified number of epochs.

        Returns
        -------
        history : dict or None
            A dictionary containing the training history (if get_History is True), otherwise None.
            The history includes the following metrics:
            - "train_MAE": List of training mean absolute error (MAE) for each epoch.
            - "train_ACC": List of training accuracy for each epoch.
            - "val_MAE": List of validation MAE for each epoch (if validation data is provided).
            - "val_ACC": List of validation accuracy for each epoch (if validation data is provided).
    '''

    def __init__(self, 
                model : nn.Module, 
                dataSet, 
                device : str = "cpu",
                batch_size : int = 64,
                dataSet_Val = None, 
                model_save_dir : str =  None):

        if(model_save_dir is None):
            raise ValueError('model_save_dir could not be None') 

        #* Initialize history to store training metrics
        self.history = {  
            "train_MAE" : [],
            "train_ACC" : [],
            "val_ACC"   : [],
            "val_MAE"   : []
            }

        #* Initialize class attributes
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
        
        if(self.dataSet_Val != None):
            self.data_loader_Val = DataLoader(
                                    self.dataSet_Val, 
                                    batch_size  = self.batch_size,
                                    num_workers = 0,
                                    shuffle = True)

    
    def getAccuracy_and_MAE(self, criterion : torch.nn.Module, data_loader : DataLoader):
        '''
            Parameters
            ----------
            data_loader : DataLoader  
                index list of the batch tensors of the dataset
            criterion :  torch.nn.Module, optional
                loss function of the model
            device : str
                Device that we use like GPU

            Returns
            -------
            Return a tuple with the accuracy of the model in the data loader and the
            model total loss in the data loader.
        '''

        #* Set the model to evaluation mode
        self.model.eval()
        model_ACC = 0
        model_MAE = 0
        size_data_loader = len(data_loader.dataset)

        #* Disable gradient computation
        with torch.no_grad(): 
            for imgs, labels in data_loader:
                labels  = labels.to(self.device)
                imgs    = imgs.to(self.device)
                outputs = self.model(imgs)

                batch_loss = criterion(outputs, labels)  
                model_MAE += batch_loss.item()*self.batch_size

                #* Get the index of the maximum predicted value
                prediction = torch.max(outputs, 1)[1]  
                #* Count the number of correct predictions
                model_ACC += (prediction == labels).sum().item()  

        model_ACC = model_ACC / size_data_loader  # Compute the accuracy
        model_MAE = model_MAE / size_data_loader

        return model_ACC, model_MAE

    def fitModel(self, 
                opt_model  : torch.optim, 
                criterion  : torch.nn.Module = torch.nn.CrossEntropyLoss(),
                num_epochs : int = 1):
        '''
            Train the model for the specified number of epochs using criterion, and opt_model.
            
            Parameters
            ----------
            opt_model : optim
                The optimization algorithm used for training.
            criterion : torch.nn.Module, optional
                The loss function used for training (default torch.nn.CrossEntropyLoss()).
            num_epochs : int, optional
                The number of epochs to train the model (default is 1).
        '''

        if num_epochs < 0:
            raise ValueError('num_epochs should be non-negative') #todo test

        sizeDataSet = len(self.data_loader.dataset) 
        self.model.to(self.device)


        for epoch in range(num_epochs):
            loop = tqdm(enumerate(self.data_loader), total = len(self.data_loader))
            self.model.train()#* Put model in training mode
            train_MAE = 0 
            train_ACC = 0 

            for batch_idx, (imgs, labels) in loop:
                imgs    =   imgs.to(self.device)
                labels  = labels.to(self.device)
                opt_model.zero_grad()
                outputs = self.model(imgs)

                #* Get the preditions and the loss for the batch
                loss       = criterion(outputs, labels)  
                train_MAE += loss.item()*self.batch_size  
                prediction = torch.max(outputs, 1)[1]  
                train_ACC += (prediction == labels).sum().item()

                #* Get gradients, and update the parameters of the model
                loss.backward()
                opt_model.step()

                #* Plot the loss and the progress bar
                loop.set_description(f"Epoch {epoch+1}/{num_epochs} process: {int((batch_idx / len(self.data_loader)) * 100)}")
                loop.set_postfix(modelLoss = loss.data.item())

            train_ACC = train_ACC / sizeDataSet
            train_MAE = train_MAE / sizeDataSet
            print(f'Epoch completed, TRAIN MAE: {train_MAE:.4f}')
            print(f'Epoch completed, TRAIN ACC: {train_ACC:.4f}')
            self.history["train_MAE"].append(train_MAE) 
            self.history["train_ACC"].append(train_ACC)

            #* Get the ACC, and MAE in val dataSet and plot them, if we have validation data.
            if(self.dataSet_Val != None):
                ACC_Val, MAE_Val = self.getAccuracy_and_MAE( 
                                                    data_loader = self.data_loader_Val, 
                                                    criterion   = criterion, 
                                                    )

                print(f'Epoch completed, VAL MAE: {(MAE_Val):.4f}')
                print(f'Epoch completed, VAL ACC: {(ACC_Val):.4f}')
                self.history["val_MAE"].append(MAE_Val)
                self.history["val_ACC"].append(ACC_Val)

            #* Save the model every epoch
            torch.save({ 
                'model_state_dict': self.model.state_dict(), 
                'optimizer_state_dict': opt_model.state_dict(), 
            }, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))
