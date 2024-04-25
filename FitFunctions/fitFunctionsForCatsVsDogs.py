
import torch
import torch.nn as nn
import os
from tqdm import tqdm



def train_modelCNN(data_loader, model, opt_model, device, 
                    data_loader_Val = None, 
                    num_epochs = 1, criterion = torch.nn.CrossEntropyLoss(), 
                    get_History = True, getVal = False,
                    model_save_dir = None):
    '''
        Fit function for a clasificator cat vs dog model. This function will 
        save the models in each epoch, and it return the history if we whant.

        Parameters
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



