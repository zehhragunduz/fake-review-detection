import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Kaydedilmiş modeli ve tokenizer'ı yükle
model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. Yorumu işle ve tahmin et
def predict(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "✅ GERÇEK" if pred == 1 else "❌ SAHTE"
    print(f"\n📝 Yorum: {text}")
    print(f"{label} yorum (%{confidence * 100:.2f} güven)\n" + "-"*60)

# 3. Kullanıcıdan yorum al
print("📢 BERTurk Yorum Sınıflandırıcıya Hoşgeldin!\nÇıkmak için 'q' yaz.\n")

while True:
    user_input = input("Yorum girin: ")
    if user_input.lower() == "q":
        print("Çıkılıyor... 💨")
        break
    predict(user_input)
