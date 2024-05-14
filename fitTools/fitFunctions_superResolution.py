
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader


class fitertImgToImg():
    '''
        This class is for train a img to img models and test it. 
        It saves the models at the end of each epoch and save the training history.

        .. warning::
            For use the trainModel method the model output, and the outPut imgs of the
            data_loader need to be compatibles using the funcion criterion 
            in the trainModel method.

        Attributes
        ----
            model : nn.Module
                The model to train.
            dataSet : Dataset
                The dataset containing the training data.
            history : dict
                A dictionary with keys "train_MAE" and "val_MAE" and the values 
                are historial lists of that value. 
            device  : str
                The environment devicedevice where we will do our calculations.
            batch_size : int, optional
                The batch size used for training (default is 64)
            dataSet_Val : Dataset, optional
                The validation data set. 
            model_save_dir : str, optional
                The path were we save the model
            training_epochs : int
                Number of training epochs the model has undergone.
            data_loader : torch.utils.data.DataLoader
                A DataLoader make with the dataSet.
            data_loader_Val : torch.utils.data.DataLoader
                A DataLoader make with the dataSet_Val.

        Methods
        -------
            getMAE(
                    data_loader : DataLoader, 
                    criterion : torch.nn.Module = torch.nn.CrossEntropyLoss()
                    ) -> float:
                Return the MAE in the 'DataLoader' using 'criterios'.

            trainModel(
                        opt_model  : torch.optim.Optimizer,  
                        criterion  : torch.nn.Module = torch.nn.CrossEntropyLoss(),
                        num_epochs : int = 1
                        ) -> None:
                Train the model using 'num_epochs', 'criterion', 'opt_model', and 
                the 'data_loader' attributes class.
            plotHistory(
                        intervalTrain : list[int] = None, 
                        intervalValidation : list[int] = None
                        )->None:
                Plot the attribute history.
            predict(
                    index : int = 0, 
                    imgPath : str = None
                    ) -> tensor:
                Compute the prediction using the model.          
    '''

    def __init__(self,
                model : nn.Module, 
                dataSet, 
                device     : str = "cpu",
                batch_size : int = 64,
                dataSet_Val = None, 
                model_save_dir : str =  None):
        '''
            Initializes a new instance of the class fiterImgToImg.

            Args:
                model : nn.Module
                    The model to train.
                dataSet : Dataset
                    The dataset containing the training data.
                    The input img and the ouPut img need to be compatible with
                    the model input and the modelOutPut respective.
                device : str
                    The environment devicedevice where we will do our calculations.
                batch_size : int, optional
                    The batch size used for training (default is 64)
                dataSet_Val : Dataset, optional
                    The validation data set. 
                model_save_dir : str, optional
                    The path were we save the model
        '''

        if(model_save_dir is None):
            raise ValueError('model_save_dir could not be None')

        #* Initialize history
        self.history = {
            "train_MAE" : [], #* int list
            "val_MAE"   : [], #* pair(int, int) list
            #todo add the PSNR, or SSIM, FID
        }

        #* Start the class attributes
        self.model       = model
        self.dataSet     = dataSet
        self.device      = device
        self.batch_size  = batch_size
        self.dataSet_Val = dataSet_Val
        self.model_save_dir  = model_save_dir
        self.training_epochs = 0

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
        else:
            self.data_loader_Val = None

    def getMAE(self,
            data_loader : DataLoader, 
            criterion : torch.nn.Module = torch.nn.CrossEntropyLoss(),
            ):
        '''
            Compute and return the MAE in 'data_loader' using 'criterion'.
            
            Args:
            -----
                data_loader : torch.utils.data.DataLoader  
                    index list of the batch tensors of the dataset
                criterion :  torch.nn.Module, optional
                    loss function of the model

            Returns
            -------
                Return a the value of MAE in 'data_loader' using 'criterion'.
        '''

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
                opt_model  : torch.optim.Optimizer, 
                criterion  : torch.nn.Module = torch.nn.CrossEntropyLoss(),
                num_epochs : int  = 1,
                getValMAE  : bool = False):
        '''
            Train the model in the device using 'num_epochs', 'criterion', 
            'opt_model' and the dataloaders class attribut.
            We need criterion(model(imgInput), imgOutPut). So the model(imgInput) and
            the imgOutPut need to be compatible in criterion i.e model(imgInput).shape
            = imgOutPut.shape ??(todo).

            Args:
            ----------
                opt_model : torch.optim.Optimizer
                    The optimization algorithm used for training.
                criterion : torch.nn.Module, optional
                    The loss function used for training (default torch.nn.CrossEntropyLoss()).
                num_epochs : int, optional
                    The number of epochs to train the model (default is 1).
        '''

        if num_epochs <= 0:
            raise ValueError('The num_epochs should be positive')
        
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
                train_MAE += loss.item()*imgInput.shape[0] #* imgInput.shape[0] = self.batch_size, but the last batch could be diferente size

                #* Get gradients, and update the parameters of the model
                loss.backward()
                opt_model.step()

                #* Plot the loss and the progress bar
                loop.set_description(f"Epoch {epoch+1}/{num_epochs} process: {int((batch_idx / len(self.data_loader)) * 100)}")
                loop.set_postfix(modelLoss = loss.data.item())
            self.training_epochs += 1
            train_MAE = train_MAE / sizeDataSet
            print(f'Epoch completed, TRAIN MAE {train_MAE:.4f}')
            self.history["train_MAE"].append(train_MAE)

            if((self.dataSet_Val is not None) and getValMAE == True):
                #* We could try a diferent criterio for the val case in the same dataset.
                val_MAE = self.getMAE(data_loader = self.data_loader_Val, criterion = criterion)
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
                    }, os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch + 1}_Train_MAE_{"{:.5f}".format(train_MAE)}.pt'))
    #TODO return the best model in MAE?

    def toggle_trainingLayers(self, layers_list : list[str], enable : bool):
        '''
            This function will enable or disable the layers in the layers_list for the 
            training. Afther enable the layers we will print all the enable layers.

            Args:
            -----
                layers_list : list[str]
                    List with the layers name that we will enable for the training.
                enable : bool
                    True if the layer will be enable, false if the layer will be disable.
        '''

        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layers_list):
                param.requires_grad = enable

        for name, param in self.model.named_parameters():
            if(param.requires_grad == True):
                print(f"{name} : {param.requires_grad}")

    def printHistorial(self, 
                        intervalTrain : list[int] = None, 
                        intervalValidation : list[int] = None):
        '''
            Plot a img with the historial values that we have.

            Args:
                intervalTrain : list[int], optional
                    The interval of training epochs that we will plot.
                intervalValidation : list[int], optional
                    The interval of training epochs that we will plot.
        '''

        if self.training_epochs == 0:
            print("self.training_epochs == 0 i.e model was not trained")
            return

        if intervalTrain is None:
            intervalTrain = [0, len(self.history['train_MAE'])]

        if intervalValidation is None:
            intervalValidation = [0, len(self.history['val_MAE'])]

        if intervalTrain[0] < 0 or intervalTrain[0] > self.training_epochs:
            raise ValueError('The intervalTrain[0] need to be in [0, training_epochs of the model)')
        if intervalTrain[1] > self.training_epochs:
            raise ValueError('The intervalTrain[1] need to be in [0, training_epochs of the model)')
        if intervalTrain[0] >= intervalTrain[1]:
            raise ValueError('We need intervalTrain[0] < intervalTrain[1]')
        if intervalValidation[0] < 0 or intervalValidation[0] > self.training_epochs:
            raise ValueError('The intervalValidation[0] need to be in [0, training_epochs of the model)')
        if intervalValidation[1] > self.training_epochs:
            raise ValueError('The intervalValidation[1] need to be in [0, training_epochs of the model)')
        if intervalValidation[0] >= intervalValidation[1]:
            raise ValueError('We need intervalValidation[0] < intervalValidation[1]')

        Epochs_values     = range(intervalTrain[0], intervalTrain[1])
        Epochs_values_Val = range(intervalValidation[0], intervalValidation[1])

        if(len(self.history['val_MAE']) != 0): #* Two img plots
            fig, (plt1) = plt.subplots(1, 1, figsize=(12, 6))
            plt1.plot(Epochs_values,   self.history['train_MAE'][intervalTrain[0]: intervalTrain[1]], marker='o', color='blue', label='train MAE')
            plt1.set_xlabel('Epoch')
            plt1.set_title('Train MAE')     
            plt1.plot(Epochs_values_Val, self.history['val_MAE'][intervalValidation[0]: intervalValidation[1]], marker='o', color='red', label='validation MAE')
            plt1.set_xlabel('Epoch')
            plt1.set_title('Validation MAE Vs Train MAE')

            #* Add legend to each subplot
            plt1.legend()
            plt1.legend()
            #* Show the plots
            plt.show()

        elif(len(self.history['train_MAE']) != 0): #* One img plot
            fig, (plt1) = plt.subplots(1, 2, figsize=(12, 6))
            plt1.plot(Epochs_values, self.history['train_MAE'][Epochs_values[0] : Epochs_values[-1]], marker='o', color='blue', label='MAE')
            plt1.set_xlabel('Epoch')
            plt1.set_ylabel('MAE')
            plt1.set_title('Train MAE')
            
            #* Add legend to each subplot
            plt.legend()
            #* Show the plots
            plt.show()
        
        else:
            print("len(self.history['val_MAE']) == 0, and \n len(self.history['train_MAE']) == 0")

    def predict(self, index : int = 0, imgPath : str = None):
        '''
            Use the model in a img, or in a attribute data_loader[index]

            Args:
            -----
                index : int
                    Index of the img in the data_loader
                imgPath : str
                    Path of the image that we will use as model input.
        '''
        #TODO
        if(imgPath is None):
            print("model(self.dataLoaders[index])")
        else:
            print("modelImgPath")

    def getDataBatch(self, index : int = 0):
        '''
            Get the a batch in data_loader_Val for do testing.
            
            Args
            ----
                index : int = 0, optional
                    The index of the tensor batch in the data loader val.
            
            Returns
            -------
                Returns a data batch of the validation data set.
        '''

        for idx, (imgInput, imgOutPut) in enumerate(self.data_loader_Val):
                if idx == index:
                    return imgInput, imgOutPut

        return None, None

class fiterU_Net(fitertImgToImg):

    def __init__(self, 
                model: nn.Module, 
                dataSet, 
                device: str = "cpu", 
                batch_size: int = 64, 
                dataSet_Val = None, 
                model_save_dir: str = None):
        super().__init__(model, dataSet, device, batch_size, dataSet_Val, model_save_dir)

    def getMAE(self,
            data_loader : DataLoader, 
            criterion : torch.nn.Module = torch.nn.CrossEntropyLoss(),
            ):
        '''
            Compute and return the MAE in 'data_loader' using 'criterion'.

            Args:
            -----
                data_loader : torch.utils.data.DataLoader  
                    index list of the batch tensors of the dataset
                criterion :  torch.nn.Module, optional
                    loss function of the model

            Returns
            -------
                Return a the value of MAE in 'data_loader' using 'criterion'.
        '''

        size_Data_loader = len(data_loader.dataset)
        model_MAE = 0

        if size_Data_loader == 0:
            raise ValueError('The data set should not be empty.')

        with torch.no_grad():
            for (imgInput, imgOutPut) in data_loader:
                imgInput    =  imgInput.to(self.device)
                imgOutPut   = imgOutPut.to(self.device, torch.long)
                modelOutPut = self.model(imgInput)

                modelOutPut = modelOutPut.view(modelOutPut.shape[0], 2, -1)
                imgOutPut = imgOutPut.view(imgOutPut.shape[0],  1, 68*68).squeeze(1)
                imgOutPut = imgOutPut.squeeze(1)
                loss = criterion(modelOutPut, imgOutPut)
                model_MAE += loss.item()*self.batch_size

        return model_MAE/size_Data_loader


    def trainModel(self,
                opt_model  : torch.optim, 
                criterion  : torch.nn.Module = torch.nn.CrossEntropyLoss(),
                num_epochs : int = 1,
                getValMAE  : bool = False):
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
                imgOutPut = imgOutPut.to(self.device, torch.long)
                opt_model.zero_grad()
                modelOutPut = self.model(imgInput)

                #* Get the batch loss and computing train MAE
                modelOutPut = modelOutPut.view(modelOutPut.shape[0], 2, -1)
                imgOutPut = imgOutPut.view(imgOutPut.shape[0],  1, 68*68).squeeze(1)
                imgOutPut = imgOutPut.squeeze(1)
                loss       =  criterion(modelOutPut, imgOutPut)
                train_MAE += loss.item()*imgInput.shape[0] #* imgInput.shape[0] = self.batch_size, but the last batch could be diferente size

                #* Get gradients, and update the parameters of the model
                loss.backward()
                opt_model.step()

                #* Plot the loss and the progress bar
                loop.set_description(f"Epoch {epoch+1}/{num_epochs} process: {int((batch_idx / len(self.data_loader)) * 100)}")
                loop.set_postfix(modelLoss = loss.data.item())

            self.training_epochs += 1
            train_MAE = train_MAE / sizeDataSet
            print(f'Epoch completed, TRAIN MAE {train_MAE:.4f}')
            self.history["train_MAE"].append(train_MAE)

            if((self.dataSet_Val is not None) and getValMAE == True):
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




