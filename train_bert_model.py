import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import preprocess_data
from data_loader import load_data
import os
import pickle

def build_model(vocab_size, input_length, output_dim=6):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='sigmoid')  # Multi-label binary classification
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load data
    train_df, _ = load_data("data/train.csv", "data/test.csv")
    X, y, tokenizer = preprocess_data(train_df)
    
    vocab_size = len(tokenizer.word_index) + 1
    input_len = X.shape[1]
    output_dim = y.shape[1]

    # Build and train model
    model = build_model(vocab_size, input_len, output_dim)
    print(model.summary())

    es = EarlyStopping(patience=2, restore_best_weights=True)

    model.fit(X, y, epochs=5, batch_size=128, validation_split=0.2, callbacks=[es])

    # Ensure the models folder exists
    os.makedirs("models", exist_ok=True)

    # Save model
    model.save("models/bert_toxicity_model.keras")
    print("Model saved to models/toxicity_model.keras")

    # Save tokenizer
    with open("models/bert_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print(" Tokenizer saved to models/tokenizer.pkl")
