# Import necessary libraries
from datasets import load_dataset
#from transformers import BertTokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd

# Load the IMDb datasetpy
dataset = load_dataset("imdb")

# Load the tokenizer for the BERT model
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# Tokenization function to be applied to each example
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization to the training and test datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Function to convert the tokenized datasets to a PyTorch format
def format_dataset(tokenized_dataset):
    # Convert to pandas DataFrame
    df = pd.DataFrame(tokenized_dataset)
    # Labels and input IDs are required for the BERT model
    labels = df['label']
    input_ids = torch.tensor(df['input_ids'].to_list())
    attention_masks = torch.tensor(df['attention_mask'].to_list())
    return input_ids, attention_masks, labels

# Format the training and test datasets
train_input_ids, train_attention_masks, train_labels = format_dataset(tokenized_datasets['train'])
test_input_ids, test_attention_masks, test_labels = format_dataset(tokenized_datasets['test'])

# Save preprocessed data
torch.save(train_input_ids, 'train_input_ids.pt')
torch.save(train_attention_masks, 'train_attention_masks.pt')
torch.save(train_labels, 'train_labels.pt')
torch.save(test_input_ids, 'test_input_ids.pt')
torch.save(test_attention_masks, 'test_attention_masks.pt')
torch.save(test_labels, 'test_labels.pt')

print("Data preparation completed successfully!")
