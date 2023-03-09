import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import csv

df = pd.read_csv('test.csv', encoding='utf-8')
X_test = df.text.tolist()
dfid = df['id']
class_names = ['0', '1']
print('size of test set: %s' % (len(X_test)))

# Tokenize the input data
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', use_fast=True)
x_test = tokenizer(X_test, padding=True, truncation=True, max_length=350, return_tensors='pt')
x_test = x_test.to('cuda')

# Create PyTorch Dataset and DataLoader
class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to('cuda') for key, val in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])

test_dataset = CustomDataset(x_test)
test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False)

# Load model
model = AutoModelForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=len(class_names))
model.to('cuda')

# Load saved model weights
model.load_state_dict(torch.load('bert_model_large_AdamW_3_0.8517060367454068.pt'))

# Prediction loop
model.eval()
with torch.no_grad():
    predictions = []
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        outputs = model(input_ids, attention_mask=attention_mask)
        _, batch_predictions = torch.max(outputs.logits, dim=1)
        predictions.extend(batch_predictions.cpu().tolist())

# Save predictions to CSV file
with open('predictionsLarge10.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'target'])
    for i, pred in enumerate(predictions):
        writer.writerow([dfid[i], pred])
