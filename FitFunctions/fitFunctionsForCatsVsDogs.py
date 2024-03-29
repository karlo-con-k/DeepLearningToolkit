
import torch
import torch.nn as nn


def train_modelCNN(data_loader, model, opt_model, device, data_loader_Val = None, num_epochs = 1, 
                    get_History = True, criterion = torch.nn.MSELoss(), getValLoos = False, test = False):

    '''
        data_loader = index list of the batch tensors of the dataset
        model       = model to fit
        opt_model   = optimization algorihm
        device      = device that will use like GPU
        history     = if true, then create a history dictionary and will append the values loss, and accuracy
        criterion   = losst function of the model
    '''

    if(test == True):
        num_epochs = 1

    if(get_History):
        history = {
                    "MAE" : [],
                    "ACC" : []
                    }
        if(getValLoos == True):
            history['MAE_Test'] = []
            
    model.to(device)
    model.train()

    sizeDataSet     = len(data_loader.dataset)          #* size of the dataSet i.e number of images
    sizeDataSetTest = len(data_loader.dataset)          #* size of the dataSetTest 
    batch_size      = data_loader.batch_size

    for epoch in range(num_epochs):
        print("Epoch = ", epoch)
        model_total_loss  = 0
        for idx, (img, label) in enumerate(data_loader):
            img        = img.to(device)
            label      = label.float().to(device)
            prediction = model(img)

            loss       = criterion(prediction, label)       #* Criterio loss batch img
            model_total_loss += loss.item()*batch_size      #* I will save the Mean Absolute Error of the dataSet (this sumad is the averance loss)
                                                            #* loss.item() for get the value of the tensor.
            opt_model.zero_grad()
            loss.backward()
            opt_model.step()
            if (idx + 1) % (sizeDataSet/10) == 0: #todo why do not get 10 prints ???
                print(f'Batch [{idx + 1}/{sizeDataSet}], Loss: = {loss.item():.4f}')

        if(history):
            history["MAE"].append(model_total_loss/sizeDataSet) #* MAE Mean Absolute Error
            history["ACC"].append(model_total_loss/sizeDataSet) #* MAE Mean Absolute Error
            #TODO define well the ACC
        

        #TODO ADD THE TEST LOOS AND TEST ACC
        if(getValLoos == True):
            model_total_loss_test = 0
            for idx, (img, label) in enumerate(data_loader_Val):
                img        = img.to(device)
                label      = label.float().to(device)
                prediction = model(img)

                loss       = criterion(prediction, label)      #* Criterio loss batch img
                model_total_loss_test += loss.item()*batch_size     #* I will save the Mean Absolute Error of the dataSet (this sumad is the averance loss)
            
            history["MAE_Test"].append(model_total_loss_test/sizeDataSetTest)
            print(f'Epoch completed, Average Test Loss: {(model_total_loss_test/sizeDataSetTest):.4f}')

        avg_loss = model_total_loss / sizeDataSet
        print(f'Epoch completed, Average Loss: {avg_loss:.4f}')


    return history
