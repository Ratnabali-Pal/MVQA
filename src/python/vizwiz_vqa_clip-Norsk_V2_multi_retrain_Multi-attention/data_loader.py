import json
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def find_most_common_answer(answers)
    Finds the most common answer from a list of answers.
    answer_counter = Counter(answers)
    most_common_answers = answer_counter.most_common()
    most_common_answer, _ = most_common_answers[0]
    return most_common_answer

def select_most_common_answers(df)
    Selects the most common answer for each question in the DataFrame.
    selected_answers = []
    for _, row in df.iterrows()
        answers = [answer[answer] for answer in row[answers]]
        selected_answer = find_most_common_answer(answers)
        selected_answers.append({answer selected_answer})
    df[[answer]] = pd.DataFrame(selected_answers)
    return df.drop([answers], axis=1)

def dataloader_json(path, test=False)
    Loads and processes the JSON data from the given path.
    with open(path, 'r') as f
        data = json.load(f)
    df = pd.DataFrame(data)
    if test
        return df
    return select_most_common_answers(df)

def encode_labels(df)
    Encodes the answer and answer_type columns.
    ans_lb = LabelEncoder()
    df['answer'] = ans_lb.fit_transform(df['answer'])
    ans_type_lb = LabelEncoder()
    df['answer_type'] = ans_type_lb.fit_transform(df['answer_type'])
    return df, ans_lb, ans_type_lb