import json
import pandas as pd
from collections import Counter

def find_most_common_answer(answers):
    """
    Finds the most common answer from a list of answers.

    Args:
        answers (list): A list of answer strings.

    Returns:
        str: The most common answer string.
    """
    answer_counter = Counter(answers)
    most_common_answers = answer_counter.most_common()
    most_common_answer, count = most_common_answers[0]
    return most_common_answer

def select_most_common_answers(df):
    """
    Selects the most common answer for each question in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a column named 'answers'.

    Returns:
        pd.DataFrame: The DataFrame with the 'answers' column replaced by a single 'answer' column.
    """
    selected_answers = []
    for idx, row in df.iterrows():
        answers = [answer["answer"] for answer in row["answers"]]
        selected_answer = find_most_common_answer(answers)
        selected_answers.append({"answer": selected_answer})

    df[["answer"]] = pd.DataFrame(selected_answers)
    return df.drop(["answers"], axis=1)

def dataloader_json(path, test=False):
    """
    Loads data from a JSON file into a pandas DataFrame.

    Args:
        path (str): The path to the JSON file.
        test (bool, optional): If True, returns the raw DataFrame without processing. Defaults to False.

    Returns:
        pd.DataFrame: The loaded and processed DataFrame.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    if test:
        return df

    return select_most_common_answers(df)