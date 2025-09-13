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