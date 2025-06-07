import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

def preprocess_data(df, max_words=20000, max_len=150):
    df['clean_comment'] = df['comment_text'].apply(clean_text)

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_comment'])
    sequences = tokenizer.texts_to_sequences(df['clean_comment'])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')

    y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    # Save tokenizer
    os.makedirs('../models', exist_ok=True)
    with open('../models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    return padded, y, tokenizer

if __name__ == "__main__":
    from data_loader import load_data
    train_df, _ = load_data("data/train.csv", "data/test.csv")
    X, y, tokenizer = preprocess_data(train_df)
    print("Preprocessing complete.")
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

