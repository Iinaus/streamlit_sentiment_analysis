import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

@st.cache_resource
def train_model():
    # Load the training data from a CSV file
    df = pd.read_csv('train.csv', encoding='latin1')

    # Remove missing values in the dataset
    df.dropna(inplace=True)

    # Define the input features (X) and target variable (y)
    x = df['selected_text']
    y = df['sentiment']

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)

    # Create a machine learning pipeline
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])

    # Train the model using the training data
    text_clf.fit(x_train, y_train)

    return text_clf

# Train or load the model (this is cached)
text_clf = train_model()

st.write("""
# Sentiment Analysis
Sentiment Analysis is a process of determining the emotional tone behind a piece 
of text. It is widely used to understand people's opinions, attitudes, or 
feelings toward a topic, product, or service based on written data. 
The analysis categorizes text into three main sentiment labels: positive, negative, neutral.

This tool uses machine learning models to analyze user input and classify it into 
these categories, providing real-time feedback in an intuitive and user-friendly interface.
""")

# Create a form widget for user input
form = st.form("user_input")
form.text_input("Enter a sentence to be evaluated:", key="user_input")
form.form_submit_button("Evaluate")

if st.session_state.user_input:
    # Predict sentiment for the entered sentence
    sentiment = text_clf.predict([st.session_state.user_input])
    sentiment_label = sentiment[0]

    # Define styling options based on sentiment labels
    sentiment_styles = {
        'positive': {'color': 'white', 'background_color': 'green'},
        'negative': {'color': 'white', 'background_color': 'red'},
        'neutral': {'color': 'black', 'background_color': 'yellow'}
    }

    # Get the appropriate style based on the predicted sentiment
    style = sentiment_styles.get(sentiment_label, {'color': 'black', 'background_color': 'gray'}) # Default style for unknown sentiment

    # Display the sentiment result with dynamic styling in the app
    st.markdown(
        f'<p style="background-color: {style["background_color"]}; color: {style["color"]}; padding: 10px;">'
        f'The sentence was evaluated as {sentiment_label}.</p>',
        unsafe_allow_html=True
    )