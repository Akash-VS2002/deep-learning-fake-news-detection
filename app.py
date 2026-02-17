import numpy as np
import pickle
import re
import nltk

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# Download stopwords if not present
nltk.download('stopwords')

# ----------------------------
# Load Model & Tokenizer
# ----------------------------
model = load_model("model/fake_news_bilstm_model.h5")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 400

stop_words = set(stopwords.words("english"))

# ----------------------------
# Cleaning Function (Same as Training)
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


# ----------------------------
# Input Validation Function
# ----------------------------
def is_valid_news(text):

    text = text.strip()

    # Empty check
    if text == "":
        return False, "⚠️ Please enter some news text."

    # Reject numbers or symbols only
    if not re.search(r"[a-zA-Z]", text):
        return False, "❌ Numbers or symbols only are not allowed."

    # Minimum length check
    words = text.split()
    if len(words) < 10:
        return False, "❌ News must contain at least 10 meaningful words."

    # Reject very short random strings
    if len(text) < 20:
        return False, "❌ Text is too short to be a valid news article."

    # Reject repetitive characters
    if len(set(text.lower())) < 5:
        return False, "❌ Invalid or repetitive text detected."

    return True, ""


# ----------------------------
# Prediction Function
# ----------------------------
def predict_news(text):

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(padded)[0][0]

    if pred >= 0.6:
        label = "Real News 🟢"
    else:
        label = "Fake News 🔴"

    confidence = float(pred)

    return label, confidence


# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    confidence = None
    error = None

    if request.method == "POST":

        news_text = request.form["news"]

        # Validate input
        valid, message = is_valid_news(news_text)

        if not valid:
            error = message
        else:
            result, confidence = predict_news(news_text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
