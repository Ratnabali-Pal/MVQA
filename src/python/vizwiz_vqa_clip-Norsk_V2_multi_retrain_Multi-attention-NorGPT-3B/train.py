# File: train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils import get_dataframes
from model import VQAModel

class VQADataset(Dataset):
    def __init__(self, dataframe, embeddings, indices):
        self.dataframe = dataframe.iloc[indices].reset_index(drop=True)
        # Filter embeddings based on the provided indices
        self.embeddings = [embeddings[i] for i in indices]
        
    def __getitem__(self, index):
        embedding = self.embeddings[index].float()
        answer = torch.tensor(int(self.dataframe.loc[index, 'answer']))
        answer_type = torch.tensor(int(self.dataframe.loc[index, 'answer_type']))
        return embedding, answer, answer_type
            
    def __len__(self):
          return len(self.dataframe)

def run_epoch(model, dataloader, optimizer, loss_fn, device, is_training=True):
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_correct_ans = 0
    total_correct_type = 0
    total_samples = 0
    
    context = torch.no_grad() if not is_training else torch.enable_grad()

    with context:
        for (data, ans, ans_type) in tqdm(dataloader, desc="Training" if is_training else "Validation"):
            data = data.squeeze(1).to(device)
            ans = ans.to(device)
            ans_type = ans_type.to(device)
            
            if is_training:
                optimizer.zero_grad()
            
            output, aux = model(data)
            
            loss_ans = loss_fn(output, ans)
            loss_type = loss_fn(aux, ans_type)
            loss_combined = loss_ans + loss_type
            
            if is_training:
                loss_combined.backward()
                optimizer.step()
            
            total_loss += loss_combined.item()
            
            # Answer Accuracy
            _, predicted_labels = torch.max(output, dim=1)
            total_correct_ans += (predicted_labels == ans).sum().item()
            
            # Type Accuracy
            _, predicted_labels_ty = torch.max(aux, dim=1)
            total_correct_type += (predicted_labels_ty == ans_type).sum().item()
            
            total_samples += ans.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc_ans = total_correct_ans / total_samples
    avg_acc_type = total_correct_type / total_samples
    avg_acc_total = (avg_acc_ans + avg_acc_type) / 2
    
    return avg_loss, avg_acc_total, avg_acc_ans, avg_acc_type

def plot_loss(train_loss, val_loss):
    """Plots training and validation loss curves."""
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    EMBEDDINGS_FILE = "model-n.pt"
    BATCH_SIZE = 64
    EPOCHS = 125
    LEARNING_RATE = 1e-3
    EARLY_STOPPING_PATIENCE = 15

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_df, ans_lb, ans_type_lb = get_dataframes()
    
    # Load pre-computed embeddings
    try:
        encodings = torch.load(EMBEDDINGS_FILE)
    except FileNotFoundError:
        print(f"Error: Embeddings file '{EMBEDDINGS_FILE}' not found.")
        print("Please run feature_extractor.py first to generate the embeddings.")
        exit()

    # --- Data Splitting ---
    indices = np.arange(len(data_df))
    train_indices, val_indices = train_test_split(indices, test_size=0.05, random_state=42)

    # --- Datasets and Dataloaders ---
    train_dataset = VQADataset(data_df, encodings, train_indices)
    val_dataset = VQADataset(data_df, encodings, val_indices)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model Initialization ---
    embedding_size = encodings[0].shape[1]
    num_classes = len(ans_lb.classes_)
    num_aux_classes = len(ans_type_lb.classes_)
    
    model = VQAModel(embedding_size, num_classes, num_aux_classes).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # --- Training Setup ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, threshold=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    training_losses, validation_losses = [], []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch+1}/{EPOCHS} | LR: {optimizer.param_groups[0]["lr"]}')
        
        train_loss, train_acc, _, _ = run_epoch(model, train_dataloader, optimizer, loss_fn, device, is_training=True)
        val_loss, val_acc, val_acc_ans, val_acc_type = run_epoch(model, val_dataloader, optimizer, loss_fn, device, is_training=False)
        
        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Avg Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}   | Avg Val Acc: {val_acc*100:.2f}% | Val Ans Acc: {val_acc_ans*100:.2f}% | Val Type Acc: {val_acc_type*100:.2f}%\n")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_vqa_model.pth')
            print("Validation loss improved. Saving model.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nValidation loss hasn't improved for {EARLY_STOPPING_PATIENCE} epochs. Early stopping.")
            break
            
    print("Training finished.")
    plot_loss(training_losses, validation_losses)