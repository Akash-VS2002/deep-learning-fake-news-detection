# 📰 Fake News Detection using Deep Learning (Flask)

This project predicts whether a news article is **Fake** or **Real** using a trained Deep Learning model.
It uses **NLP preprocessing**, **tokenization**, and **sequence padding (MAX_LEN = 400)** with the **same cleaning logic used during training** to ensure accurate predictions.

This README is written in a **simple, learning-friendly way** so you can clearly understand how the project works.

---

# 📌 Project Objective

The goal of this project is to:

✔ Take news text as input
✔ Clean and preprocess the text
✔ Convert text into numeric format
✔ Pass it to a Deep Learning model
✔ Predict whether the news is **Fake** or **Real**

---

# 🧠 Technologies Used

* Python
* TensorFlow / Keras
* Flask (Web Framework)
* NLP (Natural Language Processing)
* NLTK (Stopwords removal)

---

# ⚙️ Model Information

We use Deep Learning architectures such as:

* LSTM
* BiLSTM
* CNN (optional improvement)

The text is converted using:

* Keras Tokenizer
* Padding length = **400**

---

# 📁 Project Structure

```
FakeNewsProject/
│── app.py                 # Main Flask application
│── model.h5               # Trained deep learning model
│── tokenizer.pkl          # Saved tokenizer
│── requirements.txt       # Dependencies
│── README.md              # Documentation
│
├── templates/
│   │── index.html         # Input page
│   │── result.html        # Output page
│
├── utils/
│   └── preprocessing.py   # Text cleaning & padding functions
│
└── static/
    └── style.css          # Optional CSS
```

---

# 🧹 Text Cleaning Used (Same as Training)

We apply the same preprocessing during training and prediction to avoid errors.

Steps:

1. Convert to lowercase
2. Remove URLs
3. Remove special characters and numbers
4. Remove extra spaces
5. Remove stopwords

Example code:

```python
import re
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]

    return " ".join(words)
```

---

# 📏 Padding Configuration

We use the same padding length used during training:

```
MAX_LEN = 400
```

Example:

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

sequence = tokenizer.texts_to_sequences([text])
padded = pad_sequences(sequence, maxlen=400)
```

---

# 🚀 How to Run the Project

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Download NLTK Stopwords (Important)

Because we use stopwords, run this once:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## Step 3 — Run Application

```bash
python app.py
```
# 🖥️ How the System Works (Flow)

User Input → Cleaning → Tokenization → Padding → Model → Prediction → Result

---

# 📊 Output

The system will show:

* Fake News ❌
* Real News ✅

(Optional: Probability score)

---

# 🧪 Example Fake News

```
Scientists confirm that drinking hot water every 10 minutes kills all cancer cells instantly, according to a secret WHO report.
```

---

# ❗ Common Errors

## Error: No module named nltk

Solution:

```bash
pip install nltk
python -c "import nltk; nltk.download('stopwords')"
```

---

## Error: Model Not Found

Make sure these files exist:

```
model.h5
tokenizer.pkl
```

---

# 📌 Future Improvements

* Attention BiLSTM
* Confidence visualization
* Deploy to cloud (AWS / Render)
* API integration

---

# 👨‍💻 Learning Outcome

From this project you will learn:

✔ NLP preprocessing
✔ Deep Learning for text classification
✔ Model deployment using Flask
✔ Real-world ML workflow

---
If you want, I can also provide:

✅ Training Notebook
✅ Best Model Architecture (High Accuracy)
✅ Deployment Guide


