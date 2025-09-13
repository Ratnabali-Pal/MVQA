import os
from vizwiz_data_loader import dataloader_json
from vizwiz_visualizer import create_sunburst_chart

if __name__ == '__main__':
    # Define file paths
    train_json_path = "beng/train_beng.json"
    sunburst_output_path = "beng/plot_vizwiz.html"

    # Ensure the output directory exists
    output_dir = os.path.dirname(sunburst_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the training data
    train_df = dataloader_json(train_json_path)

    # Display the first few rows of the DataFrame
    print(train_df.head())
    print(f"DataFrame shape: {train_df.shape}")

    # Create and save the sunburst chart
    create_sunburst_chart(train_df, sunburst_output_path)
    print(f"Sunburst chart saved to {sunburst_output_path}")