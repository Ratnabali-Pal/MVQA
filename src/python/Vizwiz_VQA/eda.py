#copyable python code
# Description: Exploratory Data Analysis (EDA) of the VizWiz dataset.

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from data_loader import dataloader_json
import config

def plot_answer_and_type_histograms(df):
    """Generates histograms for 'answerable' and 'answer_type' columns."""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 6))

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

def plot_question_type_distribution(df):
    """Creates a sunburst chart for the distribution of question starting words."""
    question_counts = {
        "What": df['question'].str.startswith("What").sum(),
        "Why": df['question'].str.startswith("Why").sum(),
        "How": df['question'].str.startswith("How").sum(),
        "When": df['question'].str.startswith("When").sum(),
        "Is": df['question'].str.startswith("Is").sum(),
        "Can": df['question'].str.startswith("Can").sum(),
        "Where": df['question'].str.startswith("Where").sum(),
    }
    sunburst_data = pd.DataFrame(question_counts.items(), columns=['Question Type', 'Count'])
    fig = px.sunburst(
        sunburst_data, 
        path=['Question Type'], 
        values='Count', 
        title="Distribution of Questions Starting with What, Why, How, When, Is, Can, Where"
    )
    fig.show()

def analyze_question_sequences(df):
    """Analyzes and plots the distribution of the first six words in questions."""
    nltk.download('punkt')
    
    def first_six_words(question):
        tokens = nltk.word_tokenize(str(question).lower())
        return tokens[:6]

    first_words_list = [first_six_words(q) for q in df['question']]
    
    # Pad shorter lists with None to create a uniform structure
    padded_list = [row + [None] * (6 - len(row)) for row in first_words_list]
    
    words_df = pd.DataFrame(padded_list, columns=[f'Word{i+1}' for i in range(6)])
    words_df.fillna('', inplace=True) # Replace None with empty string for aggregation

    words_df['sequence'] = words_df.agg(' '.join, axis=1).str.strip()
    sequence_count = words_df['sequence'].value_counts().reset_index()
    sequence_count.columns = ['Sequence', 'Frequency']
    
    path_df = sequence_count['Sequence'].str.split(' ', expand=True)
    path_df.columns = [f'Word{i+1}' for i in range(path_df.shape[1])]
    
    # Ensure all 6 columns exist, adding empty ones if necessary
    for i in range(path_df.shape[1], 6):
        path_df[f'Word{i+1}'] = ''

    plot_df = pd.concat([path_df, sequence_count['Frequency']], axis=1)

    fig = px.sunburst(
        plot_df,
        path=[f'Word{i+1}' for i in range(6)],
        values='Frequency',
        color='Frequency',
        color_continuous_scale='RdYlBu',
        title='Sunburst Diagram: Distribution of First Six Words in VizWiz Questions'
    )
    fig.show()


if __name__ == '__main__':
    # Load data
    train_df = dataloader_json(config.TRAIN_JSON_PATH)
    val_df = dataloader_json(config.VAL_JSON_PATH)
    data_df = pd.concat((train_df, val_df), axis=0, ignore_index=True)

    print("Displaying combined data info:")
    data_df.info()
    
    print("\n--- Running EDA Visualizations ---")
    
    # Plot histograms
    plot_answer_and_type_histograms(data_df)
    
    # Plot question type sunburst
    plot_question_type_distribution(train_df)
    
    # Analyze and plot question sequences
    analyze_question_sequences(train_df)