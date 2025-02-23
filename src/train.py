import numpy as np
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def load_data(max_features):
    # Loading the IMDB Dataset
    max_features = 10000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test, max_length):
    # Padding the input data
    X_train = sequence.pad_sequences(X_train, maxlen=max_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_length)
    return X_train, X_test

def earlystopping():
    # Creating an instance of Earlystopping callback 
    es_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    return es_callback

def train_model(X_train, X_test, y_train, y_test):

    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=500))
    model.add(SimpleRNN(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es_callback = earlystopping()

    print('training Starting')
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[es_callback])

    # Saving the model
    print('Saving the Model')
    model.save('../Output/imdb_sentiment_analysis.h5')

    return 'Training Completed Successfully'

def initiate_training(max_features=10000, max_length=500, ):
    X_train, X_test, y_train, y_test = load_data(max_features)
    X_train, X_test = preprocess_data(X_train, X_test, max_length)
    train_status = train_model(X_train, X_test, y_train, y_test)
    return train_status