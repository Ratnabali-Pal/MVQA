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