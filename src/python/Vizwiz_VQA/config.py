Of course! Here is the Jupyter Notebook converted into multiple Python files that are suitable for a GitHub repository.

Project Structure

Here is a recommended file structure for your project on GitHub:

code
Code
download
content_copy
expand_less

.
├── requirements.txt
├── config.py
├── data_loader.py
├── eda.py
├── utils.py
└── feature_extractor.py
1. requirements.txt

This file lists the necessary packages for this project.

File Name: requirements.txt

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
#copyable python code
# It's recommended to install CLIP directly from github
# pip install git+https://github.com/openai/CLIP.git

matplotlib
numpy
pandas
torch
scikit-learn
Pillow
tqdm
plotly
nltk
openpyxl
2. config.py

This file centralizes all the file paths and configurations, making the code cleaner and easier to modify.

File Name: config.py

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
#copyable python code
# Description: Configuration file for file paths and parameters.

# Google Drive base path (adjust if necessary)
GDRIVE_BASE_PATH = "/content/gdrive/MyDrive/"

# Data paths
TRAIN_DIR = GDRIVE_BASE_PATH + "train"
VAL_DIR = GDRIVE_BASE_PATH + "val"
TEST_DIR = GDRIVE_BASE_PATH + "test"

# Annotation file paths
TRAIN_JSON_PATH = GDRIVE_BASE_PATH + "train.json"
VAL_JSON_PATH = GDRIVE_BASE_PATH + "val.json"
TEST_JSON_PATH = GDRIVE_BASE_PATH + "test.json" # Assuming a test.json exists

# Output paths
OUTPUT_DIR = "./output/"
QUESTION_EXCEL_PATH = OUTPUT_DIR + "question.xlsx"
3. data_loader.py

This script contains the functions responsible for loading and preprocessing the JSON data.

File Name: data_loader.py

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
#copyable python code
# Description: Functions for loading and preparing the VizWiz dataset.

import json
import pandas as pd
from collections import Counter

def find_most_common_answer(answers):
    """Finds the most frequently occurring answer from a list of answers."""
    answer_counter = Counter(answers)
    most_common_answer, _ = answer_counter.most_common(1)[0]
    return most_common_answer

def select_most_common_answers(df):
    """Processes the DataFrame to select the most common answer for each question."""
    selected_answers = []
    for _, row in df.iterrows():
        answers = [answer["answer"] for answer in row["answers"]]
        selected_answer = find_most_common_answer(answers)
        selected_answers.append({"answer": selected_answer})

    df[["answer"]] = pd.DataFrame(selected_answers)
    return df.drop(["answers"], axis=1)

def dataloader_json(path, is_test=False):
    """Loads a JSON annotation file into a pandas DataFrame."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)

    if is_test:
        return df

    return select_most_common_answers(df)
4. utils.py

This utility script holds helper functions for plotting, which can be reused across different parts of the project.

File Name: utils.py

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
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
5. eda.py

This script is for exploratory data analysis. It generates the visualizations seen in the notebook to better understand the data distribution.

File Name: eda.py

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
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
6. feature_extractor.py

This is the main script that performs the core task from the notebook: extracting image and text features using the CLIP model.

File Name: feature_extractor.py

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
#copyable python code
# Description: Extracts image and text features using the CLIP model.

import torch
import clip
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config
from data_loader import dataloader_json

def get_image_path(img_name, train_dir, val_dir, test_dir):
    """Constructs the full image path based on its name."""
    if "train" in img_name:
        return f"{train_dir}/{img_name}"
    elif "val" in img_name:
        return f"{val_dir}/{img_name}"
    elif "test" in img_name:
        return f"{test_dir}/{img_name}"
    else:
        return None

def main():
    """Main function to load data and extract features."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    model_clip, preprocess = clip.load("ViT-L/14", device=device)

    # Load and combine data
    train_df = dataloader_json(config.TRAIN_JSON_PATH)
    val_df = dataloader_json(config.VAL_JSON_PATH)
    data_df = pd.concat((train_df, val_df), axis=0, ignore_index=True)

    # Encode labels
    ans_lb = LabelEncoder()
    data_df['answer'] = ans_lb.fit_transform(data_df['answer'])
    ans_type_lb = LabelEncoder()
    data_df['answer_type'] = ans_type_lb.fit_transform(data_df['answer_type'])

    encodings = []

    print("Starting feature extraction...")
    for _, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        img_name = row['image']
        question = row['question']
        
        # Note: The original notebook had a path issue. This function corrects it.
        image_path = get_image_path(img_name, config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR)

        if image_path is None:
            print(f"Warning: Could not determine path for image: {img_name}")
            continue

        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            text = clip.tokenize(question).to(device)

            with torch.no_grad():
                image_encoding = model_clip.encode_image(image)
                text_encoding = model_clip.encode_text(text)

            # Concatenate image and text encodings
            combined_encoding = torch.cat([image_encoding, text_encoding], dim=-1)
            encodings.append(combined_encoding)

        except FileNotFoundError:
            print(f"Error: Image not found at path: {image_path}")
        except Exception as e:
            print(f"An error occurred with image {img_name}: {e}")

    print(f"Successfully processed and encoded {len(encodings)} samples.")
    
    # The 'encodings' list now holds the combined feature tensors.
    # From here, you can save them to a file or use them to train a model.
    # Example: all_encodings = torch.cat(encodings, dim=0)
    # torch.save(all_encodings, 'clip_features.pt')

if __name__ == '__main__':
    main()