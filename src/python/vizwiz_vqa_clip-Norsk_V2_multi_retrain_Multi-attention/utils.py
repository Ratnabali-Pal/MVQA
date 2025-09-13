import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_loss(train_loss, val_loss):
    """Plots the training and validation loss."""
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_img(path):
    """Plots an image from the given path."""
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def plot_histograms(df):
    """Plots histograms for 'answerable' and 'answer_type'."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    ax1.hist(df['answerable'], bins=10, alpha=0.5, color='black', align='mid')
    ax1.set_title('Histogram of Answerable')
    ax1.set_xlabel('Values')
    ax1.set_ylabel('Frequency')

    ax2.hist(df['answer_type'], bins=10, alpha=0.5, color='red', align='mid')
    ax2.set_title('Histogram of Answer Type')
    ax2.set_xlabel('Values')
    ax2.set_ylabel('Frequency')

    fig.tight_layout()
    plt.show()