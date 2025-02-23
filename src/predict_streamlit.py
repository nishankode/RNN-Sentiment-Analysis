import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the word index and reverse word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('Output/imdb_sentiment_analysis.h5')

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed = preprocess_text(review)
    prediction = model.predict(preprocessed)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    confidence = prediction[0][0] if sentiment == 'Positive' else 1 - prediction[0][0]
    return sentiment, confidence

# Streamlit app
st.title("IMDb Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")

review = st.text_area("Movie Review", height=150)

if st.button("Predict Sentiment"):
    if review.strip():
        sentiment, confidence = predict_sentiment(review)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
    else:
        st.write("Please enter a review to analyze.")
