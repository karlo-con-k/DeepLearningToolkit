
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import torch.nn.functional as F



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

        #TODO ADD Val ACC
        if(getValLoos == True): 
            model_total_loss_Val = 0
            for idx, (imgInput, imgOutput) in enumerate(data_loader_Val):
                #todo remove the enumerate
                imgInput        = imgInput.to(device)
                imgOutput      = imgOutput.to(device)
                prediction = model(imgInput)
                loss       = criterion(prediction, imgOutput)      
                model_total_loss_Val += loss.item()*batch_size     

            if(get_History):
                history["MAE_Val"].append(model_total_loss_Val/sizeDataSetVal)

            print(f'Epoch completed, Average Val Loss: {(model_total_loss_Val/sizeDataSetVal):.4f}')

        #* Save the model afther the epoch
        torch.save({ 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': opt_model.state_dict(), 
            }, os.path.join(model_save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

    if(get_History):
        return history
    

def getMSE(model, data_loader, criterion, batch_size, device):
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
