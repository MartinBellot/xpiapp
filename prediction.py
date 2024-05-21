import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from transformers import pipeline
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence

model = load_model('sentiment_model.h5')

def predict_sentiment(text, maxlen = 100):
    word_index = imdb.get_word_index()
    sequence = text_to_word_sequence(text)
    indexed_sequence = [word_index.get(word, 2) if word_index.get(word, 2) < 10000 else 2 for word in sequence]  # 2 = index pour les mots hors vocabulaire
    padded_sequence = pad_sequences([indexed_sequence], maxlen=maxlen)
    
    prediction = model.predict(padded_sequence)
    return 'positive' if prediction[0] > 0.5 else 'negative'

if __name__ == '__main__':
    while (True):
        text = str(input("Entrez un texte à analyser : "))
        if text == "exit":
            break
        sentiment = predict_sentiment(text)
        print(f"Le sentiment de ce texte est : {sentiment}")