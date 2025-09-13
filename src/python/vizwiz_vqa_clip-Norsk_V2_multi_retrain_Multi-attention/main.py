import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import clip
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data_loader import dataloader_json, encode_labels
from utils import plot_loss, plot_histograms
from model import Model

# --- 1. Data Loading and Preprocessing ---
train_df = dataloader_json("norsk/train_norsk.json")
val_df = dataloader_json("norsk/val_norsk.json")
data_df = pd.concat((train_df, val_df), axis=0, ignore_index=True)

# Plot histograms
plot_histograms(data_df)

# Encode labels
data_df, ans_lb, ans_type_lb = encode_labels(data_df)

# --- 2. Setup Model and Tokenizer ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-L/14", device=device)
tokenizer = AutoTokenizer.from_pretrained("ltg/nort5-large")
text_model = AutoModelForSeq2SeqLM.from_pretrained("ltg/nort5-large", trust_remote_code=True)
text_model.to(device)

print(f'Using {device}')

# --- 3. Generate Embeddings ---
encodings = []
target_length = 768
for img, question in tqdm(zip(data_df['image'], data_df['question'])):
    try:
        if "train" in img:
            image_path = f'train/{img}'
        else:
            image_path = f'val/{img}'
        
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        encoding = tokenizer(question, return_tensors="pt").to(device)
        input_tensor = encoding.input_ids
        
        output_tensor = text_model.generate(input_tensor, max_new_tokens=77, decoder_start_token_id=7, eos_token_id=8).to(device)
        
        if output_tensor.shape[1] > target_length:
            resized_encoding = output_tensor[:, :target_length]
        else:
            padding_length = target_length - output_tensor.shape[1]
            padding = torch.zeros((output_tensor.shape[0], padding_length), device=output_tensor.device)
            resized_encoding = torch.cat((output_tensor, padding), dim=1)
        
        with torch.no_grad():
            image_encoding = model_clip.encode_image(image).to(device)
            encodings.append(torch.cat([image_encoding, resized_encoding], dim=-1))
            
    except FileNotFoundError:
        print(f"Warning: Image file not found at {image_path}. Skipping.")
        # Add a placeholder encoding or handle as appropriate
        encodings.append(torch.zeros(1, 768 + target_length, device=device))


torch.save(encodings, "model-n.pt")

# --- 4. Prepare Datasets and DataLoaders ---
indices = np.arange(len(data_df))
train_indices, test_indices = train_test_split(indices, test_size=0.05, random_state=42)

train_df = data_df.iloc[train_indices].reset_index(drop=True)
test_df = data_df.iloc[test_indices].reset_index(drop=True)

class VizWizDataset(Dataset):
    def __init__(self, indices, data, embeddings):
        self.indices = indices
        self.data = data
        self.embeddings = embeddings

    def __getitem__(self, index):
        original_index = self.indices[index]
        embedding = self.embeddings[original_index].float()
        answer = torch.tensor(int(self.data.iloc[index]['answer']))
        answer_type = torch.tensor(int(self.data.iloc[index]['answer_type']))
        return embedding, answer, answer_type
            
    def __len__(self):
        return len(self.indices)

BATCH_SIZE = 64
train_dataset = VizWizDataset(train_indices, train_df, encodings)
val_dataset = VizWizDataset(test_indices, test_df, encodings)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# --- 5. Model Training ---
embedding_size = 768
classes = len(ans_lb.classes_)
aux_classes = len(ans_type_lb.classes_)
model = Model(embedding_size, classes, aux_classes).to(device)
model = nn.DataParallel(model)

def run_model(model, dataloader, val_loader, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct, total_samples = 0, 0
    total_correct_ty, total_samples_ty = 0, 0
    
    loader = dataloader if train else val_loader
    
    with torch.set_grad_enabled(train):
        for data, ans, ans_type in tqdm(loader):
            data, ans, ans_type = data.squeeze(1).to(device), ans.to(device), ans_type.to(device)
            
            if train:
                optimizer.zero_grad()

            output, aux = model(data)
            
            loss_ans = loss_fn(output, ans)
            loss_type = loss_fn(aux, ans_type)
            loss_combined = loss_ans + loss_type
            
            if train:
                loss_combined.backward()
                optimizer.step()

            total_loss += loss_combined.item()

            _, predicted_labels = torch.max(output, dim=1)
            total_correct += (predicted_labels == ans).sum().item()
            total_samples += ans.size(0)

            _, predicted_labels_ty = torch.max(aux, dim=1)
            total_correct_ty += (predicted_labels_ty == ans_type).sum().item()
            total_samples_ty += ans_type.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    accuracy_ty = total_correct_ty / total_samples_ty
    avg_accuracy = (accuracy + accuracy_ty) / 2
    
    return avg_loss, avg_accuracy, accuracy, accuracy_ty


epochs = 125
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, threshold=1e-6)
early_stopping_patience = 15
best_val_loss = float('inf')
epochs_without_improvement = 0
training_loss, validation_loss = [], []

for e in range(epochs):
    print(f'Epoch: {e+1} | LR: {optimizer.param_groups[0]["lr"]}')
    
    tr_loss, tr_avg_acc, tr_ans_acc, tr_type_acc = run_model(model, train_dataloader, val_dataloader, optimizer, train=True)
    val_loss, val_avg_acc, val_ans_acc, val_type_acc = run_model(model, train_dataloader, val_dataloader, optimizer, train=False)
    
    training_loss.append(tr_loss)
    validation_loss.append(val_loss)
    
    scheduler.step(val_loss)
    
    print(f"\nTrain Loss: {tr_loss:.4f} | AVG Train ACC: {tr_avg_acc * 100:.4f}% | Val Loss: {val_loss:.4f} | AVG Val ACC: {val_avg_acc * 100:.2f}%")
    print(f"Train ANS ACC: {tr_ans_acc * 100:.4f}% | VAL ANS ACC: {val_ans_acc * 100:.4f}% | Train TYPE ACC: {tr_type_acc * 100:.4f}% | VAL TYPE ACC: {val_type_acc * 100:.2f}%\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print(f"\nValidation loss hasn't improved for {early_stopping_patience} epochs. Early stopping.")
        break

plot_loss(training_loss, validation_loss)

# --- 6. Final Evaluation ---
test_loss, test_avg_acc, test_ans_acc, test_type_acc = run_model(model, None, val_dataloader, optimizer, train=False)
print(f"Test TYPE ACC: {test_type_acc * 100:.4f}% | Test ANS ACC: {test_ans_acc * 100:.4f}% | AVG Test ACC: {test_avg_acc * 100:.4f}% | Test Loss: {test_loss:.4f}")