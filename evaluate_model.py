import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from preprocess import preprocess_data
from data_loader import load_data
from sklearn.model_selection import train_test_split
import os

# Load data
train_df, _ = load_data("data/train.csv", "data/test.csv")

# Split into train/val
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Load tokenizer
with open("models/bert_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Preprocess validation data
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 150
X_val = tokenizer.texts_to_sequences(val_df['comment_text'].values)
X_val = pad_sequences(X_val, maxlen=MAX_LEN)
y_val = val_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

# Load model
model = tf.keras.models.load_model("models/bert_toxicity_model.keras")
y_pred = model.predict(X_val)

# Save prediction + y_true
pred_df = pd.DataFrame(y_pred, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
pred_df['id'] = val_df['id']
pred_df['true_labels'] = list(y_val)

os.makedirs("outputs", exist_ok=True)
pred_df.to_csv("outputs/bert_val_predictions.csv", index=False)
print("[✓] bert validation predictions saved.")
