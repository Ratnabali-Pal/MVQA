import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px
import pandas as pd


def plot_loss(train_loss, val_loss):
    """
    Plots the training and validation loss.

    Args:
        train_loss (list): A list of training loss values per epoch.
        val_loss (list): A list of validation loss values per epoch.
    """
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def plot_img(path):
    """
    Plots an image from a given file path.

    Args:
        path (str): The path to the image file.
    """
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def create_sunburst_chart(df, output_path):
    """
    Generates a sunburst chart from a DataFrame of questions and saves it as an HTML file.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'question' column.
        output_path (str): The file path to save the HTML chart.
    """
    questions = df['question']
    Levels = [[], [], [], [], [], []]

    for question in questions:
        words = question.split()[:4]
        for i in range(4):
            if len(words) < i + 1:
                Levels[i].append(None)
            else:
                Levels[i].append(words[i])

    chart_df = pd.DataFrame(
        dict(A=Levels[0], B=Levels[1], C=Levels[2], D=Levels[3], E=[1] * len(Levels[0]))
    )
    chart_df = chart_df.dropna()

    for col in ['A', 'B', 'C', 'D']:
        chart_df[col] = chart_df[col].apply(lambda x: x.encode('utf-8').decode('utf-8') if x else x)

    fig = px.sunburst(chart_df, path=['A', 'B', 'C', 'D'], values='E')
    fig.write_html(output_path)