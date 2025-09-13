# File: data_utils.py

import json
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def find_most_common_answer(answers):
    """Finds the most frequent answer from a list of answers."""
    answer_counter = Counter(answers)
    most_common_answer, _ = answer_counter.most_common(1)[0]
    return most_common_answer

def select_most_common_answers(df):
    """Processes a DataFrame to select the most common answer for each question."""
    selected_answers = []
    for _, row in df.iterrows():
        answers = [answer["answer"] for answer in row["answers"]]
        selected_answer = find_most_common_answer(answers)
        selected_answers.append({"answer": selected_answer})
    
    df[["answer"]] = pd.DataFrame(selected_answers)
    return df.drop(["answers"], axis=1)

def dataloader_json(path, is_test=False):
    """Loads and processes a JSON file into a pandas DataFrame."""
    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    if is_test:
        return df
    
    return select_most_common_answers(df)

def get_dataframes(train_path="norsk/train_norsk.json", val_path="norsk/val_norsk.json"):
    """Loads and concatenates the training and validation dataframes."""
    train_df = dataloader_json(train_path)
    val_df = dataloader_json(val_path)
    
    data_df = pd.concat((train_df, val_df), axis=0, ignore_index=True)
    
    # Encode labels
    ans_lb = LabelEncoder()
    data_df['answer'] = ans_lb.fit_transform(data_df['answer'])
    
    ans_type_lb = LabelEncoder()
    data_df['answer_type'] = ans_type_lb.fit_transform(data_df['answer_type'])
    
    return data_df, ans_lb, ans_type_lb