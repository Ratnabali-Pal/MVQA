# File: feature_extractor.py

import torch
from PIL import Image
from tqdm import tqdm
import clip
from transformers import AutoTokenizer, AutoModelForCausalLM

def create_embeddings(data_df, model_clip, preprocess, model_text, tokenizer, device, target_length=768):
    """
    Extracts image and text features and combines them into a single embedding.
    """
    encodings = []
    model_text.to(device)

    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Creating Embeddings"):
        img_path = row['image']
        question = row['question']
        
        # Determine image directory
        if "train" in img_path:
            image_path = f'train/{img_path}'
        elif "val" in img_path:
            image_path = f'val/{img_path}'
        else:
            image_path = f'test/{img_path}'

        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            continue
            
        # Encode Image with CLIP
        with torch.no_grad():
            image_encoding = model_clip.encode_image(image).to(device)

        # Encode Question with NorGPT-3B
        encoding = tokenizer(question, return_tensors="pt")
        input_tensor = encoding.input_ids.to(device)
        
        output_tensor = model_text.generate(
            input_tensor, 
            max_new_tokens=77, 
            decoder_start_token_id=tokenizer.eos_token_id, 
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        ).to(device)
        
        # Pad or truncate text encoding
        if output_tensor.shape[1] > target_length:
            resized_encoding = output_tensor[:, :target_length]
        else:
            padding_length = target_length - output_tensor.shape[1]
            padding = torch.zeros((output_tensor.shape[0], padding_length), device=device)
            resized_encoding = torch.cat((output_tensor, padding), dim=1)
        
        # Combine embeddings
        combined_encoding = torch.cat([image_encoding, resized_encoding], dim=-1)
        encodings.append(combined_encoding.cpu())

    return encodings

if __name__ == '__main__':
    from data_utils import get_dataframes
    import os
    
    # --- Configuration ---
    EMBEDDINGS_FILE = "model-n.pt"
    
    # --- Load Data ---
    data_df, _, _ = get_dataframes()
    
    # --- Check for existing embeddings ---
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"Embeddings file '{EMBEDDINGS_FILE}' already exists. Skipping feature extraction.")
    else:
        print("Embeddings file not found. Starting feature extraction...")
        # --- Setup Models ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using device: {device}')
        
        clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
        
        text_model_name = 'NorGLM/NorGPT-3B'
        text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        text_model = AutoModelForCausalLM.from_pretrained(text_model_name, trust_remote_code=True)
        
        # --- Generate and Save Embeddings ---
        all_encodings = create_embeddings(data_df, clip_model, clip_preprocess, text_model, text_tokenizer, device)
        torch.save(all_encodings, EMBEDDINGS_FILE)
        print(f"Embeddings saved to '{EMBEDDINGS_FILE}'")