import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define label columns
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
THRESHOLD = 0.5  # threshold for converting probabilities to 0/1

# Load labeled train data and split for validation
train_df = pd.read_csv("data/train.csv")
_, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
y_true = val_df[labels].values

# Prediction files for each model
model_files = {
    "LSTM": "outputs/lstm_val_predictions.csv",
    "CNN": "outputs/cnn_val_predictions.csv",
    "BERT": "outputs/bert_val_predictions.csv"
}

# Evaluate each model
for model_name, filepath in model_files.items():
    print(f"\n====== {model_name} Model Evaluation ======")

    try:
        # Load predicted probabilities
        pred_df = pd.read_csv(filepath)
        y_prob = pred_df[labels].values

        # Binarize predictions using threshold
        y_pred = (y_prob >= THRESHOLD).astype(int)

        # Compute metrics
        print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    except FileNotFoundError:
        print(f"[Warning] Prediction file not found for {model_name} model: {filepath}")
    except Exception as e:
        print(f"[Error] Could not evaluate {model_name} model due to: {e}")
