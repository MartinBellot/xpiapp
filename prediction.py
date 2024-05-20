import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from transformers import pipeline

# Définir des paramètres
NUM_WORDS = 10000  # Nombre de mots les plus fréquents à conserver
MAXLEN = 100  # Longueur maximale des séquences
EMBEDDING_DIM = 128  # Dimension de l'embedding
LSTM_UNITS = 128  # Nombre de neurones LSTM
BATCH_SIZE = 32  # Taille des lots
EPOCHS = 5  # Nombre d'époques
MODEL_PATH = 'sentiment_model.h5'  # Chemin pour sauvegarder le modèle

# Vérifie si le modèle existe déjà
if not os.path.exists(MODEL_PATH):
    print("1. Chargement et prétraitement des données...")

    # Charger les données
    print("  - Chargement des données IMDB...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
    print(f"  - Nombre de critiques d'entraînement : {len(x_train)}")
    print(f"  - Nombre de critiques de test : {len(x_test)}")

    # Prétraiter les données
    print("  - Prétraitement des séquences...")
    x_train = pad_sequences(x_train, maxlen=MAXLEN)
    x_test = pad_sequences(x_test, maxlen=MAXLEN)
    print(f"  - Forme des données d'entraînement : {x_train.shape}")
    print(f"  - Forme des données de test : {x_test.shape}")

    print("\n2. Création du modèle d'apprentissage profond...")

    # Créer le modèle
    model = Sequential([
        Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAXLEN),
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(0.2),
        LSTM(LSTM_UNITS // 2),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    print("\n3. Entraînement du modèle...")

    # Entraîner le modèle
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    print("\n4. Sauvegarde du modèle entraîné...")
    model.save(MODEL_PATH)
    print(f"  - Modèle sauvegardé sous {MODEL_PATH}")

else:
    print(f"Modèle trouvé sous {MODEL_PATH}, chargement du modèle...")

# Charger le modèle sauvegardé
model = load_model(MODEL_PATH)

print("\n5. Évaluation du modèle...")

# Charger les données pour l'évaluation
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
x_test = pad_sequences(x_test, maxlen=MAXLEN)

# Évaluer le modèle
loss, accuracy = model.evaluate(x_test, y_test)
print(f"  - Précision sur les données de test : {accuracy * 100:.2f}%")

print("\n6. Prédiction du sentiment d'un texte...")

# Fonction de prédiction du sentiment
def predict_sentiment(text, model, word_index, maxlen):
    # Prétraiter le texte
    from tensorflow.keras.preprocessing.text import text_to_word_sequence
    sequence = text_to_word_sequence(text)
    indexed_sequence = [word_index.get(word, 2) for word in sequence]  # 2 = index pour les mots hors vocabulaire
    padded_sequence = pad_sequences([indexed_sequence], maxlen=maxlen)
    
    # Prédire
    prediction = model.predict(padded_sequence)
    return 'positive' if prediction[0] > 0.5 else 'negative'

# Exemple d'utilisation
word_index = imdb.get_word_index()
text = "This movie was fantastic!"
print(f"  - Texte : {text}")
print(f"  - Sentiment prédit : {predict_sentiment(text, model, word_index, MAXLEN)}")

print("\n7. Utilisation d'un modèle préentraîné avec Hugging Face Transformers...")

# Créer un pipeline pour l'analyse des sentiments
classifier = pipeline('sentiment-analysis')

# Prédire le sentiment d'un texte
text = "I love using transformers library!"
result = classifier(text)

print(f"  - Texte : {text}")
print(f"  - Sentiment prédit : {result}")
