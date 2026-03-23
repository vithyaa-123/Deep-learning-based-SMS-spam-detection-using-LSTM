# ----------------------------------------------------------
# SMS Spam Detection using LSTM
# Author: Enapa Jeeva Sri
# ----------------------------------------------------------

import kagglehub
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------------------------------
# STEP 1: Download dataset from Kaggle
# ----------------------------------------------------------
print("📥 Downloading dataset...")
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("✅ Dataset downloaded at:", path)

# Load dataset file
data_path = f"{path}/spam.csv"
df = pd.read_csv(data_path, encoding='latin-1')

# ----------------------------------------------------------
# STEP 2: Clean and prepare data
# ----------------------------------------------------------
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['message'] = df['message'].apply(clean_text)

# Encode labels (ham=0, spam=1)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

print("\n✅ Data Sample:")
print(df.head())

# ----------------------------------------------------------
# STEP 3: Train/test split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# STEP 4: Tokenize text
# ----------------------------------------------------------
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# ----------------------------------------------------------
# STEP 5: Compute class weights (for imbalance)
# ----------------------------------------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("\n⚖️ Class Weights:", class_weights)

# ----------------------------------------------------------
# STEP 6: Build LSTM model
# ----------------------------------------------------------
model = Sequential([
    Embedding(max_words, 64, input_length=max_len),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ----------------------------------------------------------
# STEP 7: Train model
# ----------------------------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

print("\n🚀 Training model...")
history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_pad, y_test),
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ----------------------------------------------------------
# STEP 8: Evaluate model
# ----------------------------------------------------------
loss, acc = model.evaluate(X_test_pad, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

y_pred = (model.predict(X_test_pad) > 0.5).astype(int)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ----------------------------------------------------------
# STEP 9: Test custom messages
# ----------------------------------------------------------
sample_sms = [
    "Congratulations! You have won a free iPhone!",
    "Hey, are we meeting for lunch today?",
    "Free entry in 2 a weekly competition to win FA Cup tickets!",
    "Your order has been shipped and will arrive tomorrow."
]

sample_seq = tokenizer.texts_to_sequences(sample_sms)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding='post')
predictions = (model.predict(sample_pad) > 0.4).astype(int)  # lower threshold a bit

print("\n📩 Sample Predictions:")
for i, msg in enumerate(sample_sms):
    label = "Spam" if predictions[i] == 1 else "Not Spam"
    print(f"Message: {msg}\nPredicted: {label}\n")
    print(f"Message: {msg}\nPredicted: {label}\n")