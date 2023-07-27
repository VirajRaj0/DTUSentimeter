import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    text = ' '.join(filtered_sentence)
    return text


def main():
    # Preprocess the text data
    df = pd.read_csv('data.csv')

    df['preprocessed_text'] = df['post'].apply(preprocess_text)

    # Create an instance of the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Function to get sentiment labels using VADER
    def get_sentiment_label(text):
        scores = analyzer.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return 'positive'
        elif scores['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    # Apply VADER sentiment analysis to the preprocessed text in the DataFrame 'df'
    df['sentiment_label'] = df['preprocessed_text'].apply(get_sentiment_label)

    # Calculate DTU's overall sentiment score
    overall_sentiment_score = df['sentiment_label'].value_counts(normalize=True)['positive']

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['preprocessed_text'])
    word_index = tokenizer.word_index

    # Convert text to sequences and pad them to a fixed length
    X_sequences = tokenizer.texts_to_sequences(df['preprocessed_text'])
    max_length = 100
    X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_padded, df['sentiment'], test_size=0.2, random_state=42)

    # Build and train an LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(word_index) + 1, 100, input_length=max_length),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    y_pred = model.predict_classes(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Print the results
    print("DTUSentimeter — Analyzes the overall sentiment of DTU on LinkedIn — [Github] — May 2023")
    print(f"Trained LSTM model with Python, sk-learn, TensorFlow, Keras, achieving {accuracy * 100:.2f}% accuracy improvement.")
    print(f"DTU's overall sentiment score: {overall_sentiment_score * 100:.2f}% positive.")

if __name__ == "__main__":
    main()
