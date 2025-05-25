import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Modeli yükle
model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_with_score(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()

    sahte_prob = probs[0].item() * 100
    gercek_prob = probs[1].item() * 100

    print(f"\n📝 Yorum: {text}")
    print(f"🟠 SAHTE olasılığı: %{sahte_prob:.2f}")
    print(f"🟢 GERÇEK olasılığı: %{gercek_prob:.2f}")

    if sahte_prob > gercek_prob:
        print("⛔ Bu yorum büyük ihtimalle SAHTE!\n")
    else:
        print("✅ Bu yorum büyük ihtimalle GERÇEK.\n")

# 🔁 Sürekli çalışsın
if __name__ == "__main__":
    print("🔍 Sahte Yorum Tespit Sistemi\nÇıkmak için 'q' yaz ve Enter'a bas.\n")

    while True:
        text = input("💬 Yorum girin: ")
        if text.strip().lower() == "q":
            print("👋 Görüşürüz tatlım!")
            break
        elif text.strip() == "":
            print("⚠️ Lütfen boş yorum girmeyin.\n")
        else:
            predict_with_score(text)
