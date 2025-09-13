# eda.py

import pandas as pd
from data_loader import load_data
import config

def perform_eda():
    """
    Performs basic exploratory data analysis on the dataset.
    """
    # Load the data
    data = load_data(config.TRAIN_DATA_PATH)
    
    # Convert to a pandas DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    print("Dataset Head:")
    print(df.head())
    
    print("\nDataset Info:")
    df.info()
    
    print("\nNumber of questions:", len(df))

if __name__ == '__main__':
    perform_eda()