from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import pandas as pd

DATA_COLUMN = 'text'
LABEL_COLUMN = 'label'

df = pd.read_csv('data.csv')

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

train_data = df_train[DATA_COLUMN]
train_labels = df_train[LABEL_COLUMN]
test_data = df_test[DATA_COLUMN]
test_labels = df_test[LABEL_COLUMN]


MAXLEN = 100  # Longueur maximale des séquences
BATCH_SIZE = 32  # Taille des lots
EPOCHS = 2  # Nombre d'époques
MODEL_PATH = 'sentiment_model.h5'
VOCAB_SIZE = 10000  # Taille du vocabulaire

tokenizer = Tokenizer(num_words=VOCAB_SIZE)

def convert_data_to_sequences(train_data, test_data):
    tokenizer.fit_on_texts(train_data)

    train_sequences = tokenizer.texts_to_sequences(train_data)
    test_sequences = tokenizer.texts_to_sequences(test_data)

    train_padded = pad_sequences(train_sequences, maxlen=MAXLEN)
    test_padded = pad_sequences(test_sequences, maxlen=MAXLEN)

    return train_padded, train_labels, test_padded, test_labels

def build_model():
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, 32, input_length=MAXLEN))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

train_padded, train_labels, test_padded, test_labels = convert_data_to_sequences(train_data, test_data)
model = build_model()
model.fit(train_padded, train_labels, epochs=EPOCHS, validation_data=(test_padded, test_labels), batch_size=BATCH_SIZE)
model.save(MODEL_PATH)