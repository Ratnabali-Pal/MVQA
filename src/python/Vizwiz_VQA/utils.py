#copyable python code
# Description: Utility functions for plotting and visualization.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_loss(train_loss, val_loss):
    """Plots training and validation loss curves."""
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_img(path):
    """Displays an image from a given file path."""
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()