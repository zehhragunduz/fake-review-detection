# 🧠 FakeReviewDetection - Sahte Yorum Tespiti Projesi

Bu proje, e-ticaret sitelerinde paylaşılan kullanıcı yorumlarını analiz ederek sahte (spam, bot, manipülasyon içerikli) yorumları tespit etmeyi amaçlar. Makine öğrenmesi, derin öğrenme ve transformer tabanlı modeller birlikte kullanılmıştır.

---

## 📁 Proje Yapısı

fake-review-detection/
├── data/ # Veri dosyaları (.csv/.xlsx)
├── models/ # Eğitilen model dosyaları (.pt/.pkl)
├── notebooks/ # Colab not defterleri
├── scripts/ # Python script dosyaları (eğitim & test)
├── train.py # Ana başlatıcı dosya (model seçimi menüsü içerir)
├── requirements.txt # Gerekli kütüphaneler listesi
├── .gitignore / LICENSE # Geliştirici dosyaları
└── README.md # Proje tanıtımı


---

## 📚 Kullanılan Veri

- Orijinal veri İngilizce sahte/gerçek yorumlardan oluşmaktadır.
- Tüm yorumlar Türkçeye çevrilmiş, veri temizliği ve dilsel sadeleştirme yapılmıştır.
- Sonuç olarak `translated_2000_reviews.xlsx` dosyası elde edilmiştir.
- Veri, `data/` klasörü altındadır.

---

## 🔍 Kullanılan Modeller

| Tür               | Algoritmalar / Yöntemler                   | Açıklama                           |
|--------------------|---------------------------------------------|-------------------------------------|
| Makine Öğrenmesi   | Logistic Regression, SVM, Naive Bayes       | TF-IDF vektörleştirme ile           |
| Derin Öğrenme      | RNN, BiLSTM                                  | Tokenizer + Embedding + LSTM       |
| Transformer        | BERTurk (dbmdz/bert-base-turkish-cased)     | HuggingFace kullanılarak            |

---

## 🛠️ Kurulum ve Kullanım

### Gereksinimler:

```bash
pip install -r requirements.txt


Model çalıştırmak için:

python train.py
Açılan menüden çalıştırmak istediğiniz modeli seçebilirsiniz:

1 → Klasik ML modelleri

2 → RNN

3 → BiLSTM

4 → BERTur

📊 Sonuçlar
Model	Accuracy	F1 Score
SVM	0.79	0.78
BiLSTM	0.81	0.80
BERTurk	0.87	0.86

🧑‍🤝‍🧑 Ekip
İsim	Sorumluluklar
Zehra Gündüz	Veri temizleme, klasik ML modelleri, Git yönetimi
Yusuf Yıldız	Derin öğrenme (RNN, BiLSTM), BERT uygulaması, testler

📄 Lisans
Bu proje MIT lisansı ile lisanslanmıştır.
Lisans detayları için LICENSE dosyasına göz atabilirsiniz.

🏫 Akademik Bilgi
Bu proje, Fırat Üniversitesi bünyesinde yürütülmüştür.



