import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

# Constants
MAX_LEN = 150
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load model and tokenizer
model = tf.keras.models.load_model("models/cnn_toxicity_model.keras")
with open("models/cnn_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# UI Layout
st.title("🧠 Toxic Comment Detector")
st.markdown("Enter a comment or upload a file to detect toxic categories.")

# Single Comment Prediction
st.header("📌 Predict Single Comment")
user_input = st.text_area("Enter your comment here")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0]

    st.subheader("Prediction Results:")
    for label, score in zip(LABELS, prediction):
        st.write(f"**{label}**: {'✅' if score > 0.5 else '❌'} ({score:.2f})")

# Bulk Prediction via CSV
st.header("📂 Bulk Predict from CSV")
file = st.file_uploader("Upload a CSV file with a 'comment_text' column", type=["csv"])

if file:
    df = pd.read_csv(file)
    if 'comment_text' not in df.columns:
        st.error("CSV must contain a 'comment_text' column.")
    else:
        sequences = tokenizer.texts_to_sequences(df['comment_text'].values)
        padded = pad_sequences(sequences, maxlen=MAX_LEN)
        preds = model.predict(padded)

        results_df = pd.DataFrame(preds, columns=LABELS)
        output = pd.concat([df[['comment_text']], results_df], axis=1)

        st.write("✅ Predictions:")
        st.dataframe(output.head(10))

        csv = output.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Predictions", csv, "bulk_predictions.csv", "text/csv")
