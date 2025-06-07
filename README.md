# Comment-Toxicity-Detection
This project detects toxic comments in text using deep learning models like CNN, LSTM, and BERT. It includes data preprocessing, model training and evaluation, and a deployed Streamlit app for real-time and bulk toxicity prediction.

🧠 Toxic Comment Detection using Deep Learning
This project focuses on detecting toxic online comments using advanced deep learning models. The goal is to classify user comments into multiple categories like toxic, severe_toxic, obscene, threat, insult, and identity_hate, helping to build safer online platforms.

📌 Project Highlights
Multi-label classification task
Built and compared 3 deep learning models: LSTM, CNN, and BERT
Evaluated models using precision, recall, f1-score, and other metrics
Deployed best-performing model (CNN) in a Streamlit web app for real-time predictions
🛠️ Tech Stack
Python 3.x
TensorFlow / Keras
Scikit-learn
Pandas / NumPy
Streamlit (for deployment)
Matplotlib / Seaborn (for visualization)
🔄 Project Workflow
1. Data Loading & Preprocessing
Loaded labeled dataset from train.csv
Text cleaning: lowercasing, removing special characters, punctuation
Tokenization and padding of sequences (MAX_LEN = 150)
Split dataset into train and validation sets (80/20 split)
2. Model Development
Developed 3 separate models:
LSTM Model: For sequence-based learning
CNN Model: For spatial pattern detection in text sequences
BERT Model: Transformer-based contextual model
3. Training & Evaluation
All models trained on multi-label classification using sigmoid activation and binary cross-entropy loss
Used early stopping to prevent overfitting
Generated validation predictions and evaluated using classification_report
4. Model Comparison
Evaluated all three models using metrics like micro/macro precision, recall, and f1-score
CNN Model was chosen for deployment based on performance and efficiency
5. Deployment using Streamlit
Developed a user-friendly Streamlit web interface
Supports:
Real-time single comment prediction
Bulk prediction from CSV files
Outputs downloadable CSV with predictions
📁 Project Structure
Toxicity_Detection/ │ ├── data/ │ └── train.csv │ ├── models/ │ ├── cnn_toxicity_model.keras │ └── cnn_tokenizer.pkl │ ├── outputs/ │ ├── cnn_val_predictions.csv │ └── model evaluation reports │ ├── scripts/ │ ├── preprocess.py │ ├── train_lstm.py │ ├── train_cnn.py │ ├── train_bert.py │ ├── evaluate_model.py │ └── compare_model.py │ ├── app/ │ └── streamlit_app.py │ ├── requirements.txt └── README.md

🚀 Running the Streamlit App
🔧 Install Dependencies
pip install -r requirements.txt

streamlit run app/streamlit_app.py
