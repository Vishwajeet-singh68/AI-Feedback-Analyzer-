import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import joblib
import os

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Step 1: Load and prepare data
# -------------------------
data_path = os.path.join("data", "cleaned_feedback.csv")
data = pd.read_csv(data_path)

# Encode labels
label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
data['label'] = data['Sentiment'].str.lower().map(label_map)

# Drop rows with invalid sentiments
data = data.dropna(subset=['label', 'clean_text'])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['clean_text'].tolist(),
    data['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# -------------------------
# Step 2: Tokenization
# -------------------------
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer)
val_dataset = FeedbackDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# -------------------------
# Step 3: Model setup
# -------------------------
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# -------------------------
# Step 4: Training loop
# -------------------------
epochs = 2
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Average Training Loss: {total_loss / len(train_loader):.4f}")

# -------------------------
# Step 5: Evaluation
# -------------------------
model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        preds.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')

print(f"\n✅ Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# -------------------------
# Step 6: Save the trained model
# -------------------------
model_save_path = os.path.join("models", "sentiment_model.pkl")
os.makedirs("models", exist_ok=True)
joblib.dump(model, model_save_path)

print(f"\n✅ Model saved to {model_save_path}")
