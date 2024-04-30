import matplotlib.pyplot as plt
import numpy as np

def plot_img_tensor(tensor,
                    plot_channels : list[int] = [0,1,2],
                    title : str = "my title"):
    
    # print(max(plot_channels), tensor.shape[0])
    if(max(plot_channels) >= tensor.shape[0] or min(plot_channels) < 0):
        raise ValueError('this channes are not in the tensor') 

    image_array = tensor[plot_channels].detach().cpu().numpy()
    image_array = np.transpose(image_array, (1, 2, 0))
    print(image_array.shape)
    plt.imshow(image_array)
    plt.axis('off')  # Turn off axis
    plt.title(title)
    plt.show()