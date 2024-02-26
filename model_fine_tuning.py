import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import numpy as np
import time
import random
import os

# Function to set seed for reproducibility
def set_seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

# Set the seed for reproducibility
set_seed()

# Load the pre-processed data
train_input_ids = torch.load('train_input_ids.pt')
train_attention_masks = torch.load('train_attention_masks.pt')
train_labels = torch.load('train_labels.pt')
train_labels = torch.tensor(train_labels, dtype=torch.long)

test_input_ids = torch.load('test_input_ids.pt')
test_attention_masks = torch.load('test_attention_masks.pt')
test_labels = torch.load('test_labels.pt')
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Convert the datasets into TensorDatasets
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# Adjust batch size to fit your system's memory
batch_size = 4  # Reduced batch size to accommodate limited resources

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=batch_size
)

# Load DistilBertForSequenceClassification
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2  # Adjust based on the number of labels
)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Define the number of epochs and the scheduler
epochs = 1
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Function for calculating accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training loop
for epoch_i in range(0, epochs):
    print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
    print('Training...')
    
    total_train_loss = 0
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        if step % 10 == 0 and not step == 0:
            print(f'Batch {step} of {len(train_dataloader)}')

        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

        model.zero_grad()        
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f'Average Training Loss: {avg_train_loss:.2f}')

# Saving the model
model_save_path = 'fine_tuned_distilbert_model.bin'
torch.save(model.state_dict(), model_save_path)

print("Fine-tuning completed successfully!")
