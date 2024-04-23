
import torch
import torch.nn as nn
import os
from tqdm import tqdm



def train_modelCNN(data_loader, model, opt_model, device, 
                    data_loader_Val = None, 
                    num_epochs = 1, criterion = torch.nn.CrossEntropyLoss(), 
                    get_History = True, getValLoos = False,
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
        getValLoos : bool, optional 
            if we whant to get the loos in the validation dataset during the training  
        model_save_dir : str, optional
            directory to save the trained model (default is None)

        Returns
        -------
            history of the training process, if get_History is True, otherwise None
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
    sizeDataSet = len(data_loader.dataset)              #* size of the dataSet i.e number of images
    
    if(data_loader_Val is not None):
        sizeDataSetVal = len(data_loader_Val.dataset)   #* size of the dataSetVal 
    else:
        sizeDataSetVal = 1

    batch_size  = data_loader.batch_size


    for epoch in range(num_epochs):

        #* This will be the Mean Absolute Error of the dataSet 
        model_total_loss  = 0
        loop = tqdm(enumerate(data_loader), total = len(data_loader))
        for batch_idx, (img, label) in loop:
            img        = img.to(device)
            label      = label.to(device) #! label = label.float().to(device) 
            opt_model.zero_grad()
            prediction = model(img)
            loss  = criterion(prediction, label)
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
            for idx, (img, label) in enumerate(data_loader_Val):
                img        = img.to(device)
                label      = label.to(device)
                prediction = model(img)
                loss       = criterion(prediction, label)      
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


def getAccuracy(model, data_loader, device):
    '''
        Parameters
        -----------
        data_loader : DataLoader  
            index list of the batch tensors of the dataset
        model : nn.Module
            model to fit
        device : str
            Device that we use like GPU

        Returns
        -------
        Return the accuracy of the model in the data loader
    '''

    #Todo test this function 

    correct = 0
    total = 0
    #* Set the model to evaluation mode
    model.eval()

    with torch.no_grad(): #* Disable gradient computation
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            #* Get the index of the maximum predicted value
            _, predicted = torch.max(outputs, 1)  
            #* Increment the total count by the batch size
            total   += labels.size(0)
            #* Count the number of correct predictions
            correct += (predicted == labels).sum().item()  
    
    accuracy = correct / total  # Compute the accuracy
    return accuracy