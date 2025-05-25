# main.py — Hatasız, checkpoint içermeyen, eğitimi bitince modeli kaydeden HALİ

import os
import shutil
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch

# 1. Veriyi oku
df = pd.read_csv("translated_2000_reviews.csv")
df = df[["translated_text", "label"]].dropna()
X_train, X_test, y_train, y_test = train_test_split(df["translated_text"], df["label"], test_size=0.2, random_state=42)

# 2. Dataset formatına çevir
train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
test_dataset = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

# 3. Tokenizer
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# 4. Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. Trainer ayarları (checkpoint kapalı!)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="no",
    logging_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    overwrite_output_dir=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 7. Eğitim
trainer.train()

# 8. Eğitim sonrası değerlendirme
print("\n📊 Eğitim tamamlandı. Test sonuçları:")
metrics = trainer.evaluate()
print(f"✅ Test Accuracy: {metrics['eval_accuracy']:.4f}")
print(f"✅ F1 Score: {metrics['eval_f1']:.4f}")

# 9. Eğitilen modeli düzgün şekilde kaydet
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
print("\n💾 Model başarıyla './saved_model' klasörüne kaydedildi.")