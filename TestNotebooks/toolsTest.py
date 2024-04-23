import matplotlib.pyplot as plt
import numpy as np

def plot_img_tensor(tensor, title = "Mi gallo"):
    image_array = tensor.detach().cpu().numpy()
    image_array = np.transpose(image_array, (1, 2, 0))
    plt.imshow(image_array)
    plt.axis('off')  # Turn off axis
    plt.title(title)
    plt.show()