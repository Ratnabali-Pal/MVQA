# data_loader.py

import json

def load_data(file_path):
    """
    Loads data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries loaded from the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, file_path):
    """
    Saves data to a JSON file.

    Args:
        data (list): The data to be saved.
        file_path (str): The path to the output JSON file.
    """
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)