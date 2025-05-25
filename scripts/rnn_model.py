# rnn_model.py — Basit bir RNN modeli ile sahte/gercek yorum sınıflandırması

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Veriyi yükle
df = pd.read_csv("translated_2000_reviews.csv")
df = df[["translated_text", "label"]].dropna()

X = df["translated_text"].values
y = df["label"].values

# 2. Train/test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TF-IDF vektörleştir
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# 4. Torch Dataset
class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ReviewDataset(X_train_tfidf, y_train)
test_dataset = ReviewDataset(X_test_tfidf, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 5. RNN Modeli
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, seq_len=1, input_dim]
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # son zaman adımı
        return self.fc(out)

# 6. Modeli oluştur ve eğit
input_dim = X_train_tfidf.shape[1]
model = RNNClassifier(input_dim=input_dim, hidden_dim=64, output_dim=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 7. Test
model.eval()
preds = []
true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        pred_labels = torch.argmax(outputs, dim=1).cpu().numpy()
        preds.extend(pred_labels)
        true.extend(y_batch.numpy())

print("\n📊 Sınıflandırma Raporu (RNN):")
print(classification_report(true, preds, target_names=["SAHTE", "GERÇEK"]))
