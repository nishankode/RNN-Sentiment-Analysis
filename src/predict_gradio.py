import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import gradio as gr

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


def gradio_predict(review):
    sentiment, confidence = predict_sentiment(review)
    return f"Sentiment: {sentiment}\nConfidence: {confidence:.2f}"

iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(lines=5, label="Enter Movie Review"),
    outputs=gr.Textbox(label="Prediction"),
    title="IMDb Sentiment Analysis",
    description="Enter a movie review to predict its sentiment (Positive/Negative).",
    examples=[
        ["The movie was fantastic! I loved every moment."],
        ["The movie was bad! The acting was terrible and the plot was unoriginal."]
    ]
)

if __name__ == "__main__":
    iface.launch()
