import matplotlib.pyplot as plt
import numpy as np

def plot_img_tensor(tensor,
                    plot_channels : list[int] = [0,1,2],
                    title    : str = "my title",
                    localplt : plt.subplots = plt.subplots(1, 1, figsize=(12, 6))):
    '''
        Function for plot a img using a tensor. 
        The tensor shape need to be [Chanels, Height, Width], and
        plot_channels sub list of [0, Chanels]
        
        Args:
            tensor : torch.Tensor
                The img for plot in a tensor.
            plot_channels : list[int] 
                The channels for the plot
            localplt : plt.subplots, optional
                The enviroment to plot the tensor img
    '''
    
    if(max(plot_channels) >= tensor.shape[0] or min(plot_channels) < 0):
        print("tensor.shape  = ", tensor.shape)
        print("plot_channels = ", plot_channels)
        raise ValueError('this channes are not in the tensor')

    image_array = tensor[plot_channels].detach().cpu().numpy()
    image_array = np.transpose(image_array, (1, 2, 0))
    print("image_array.shape = ", image_array.shape)
    
    #TODO test
    print(localplt.ax)
    localplt.imshow(image_array)
    localplt.axis('off')  # Turn off axis
    localplt.title(title)
    localplt.show()
