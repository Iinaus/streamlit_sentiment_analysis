import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

df = pd.read_csv('train.csv', encoding='latin1')
df.isnull().sum()
df.dropna(inplace=True)

x = df['selected_text']
y = df['sentiment']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

text_clf.fit(x_train, y_train)

st.write("""
# Sentiment Analysis
""")

st.text_input("Enter a sentence:", key="user_input")

if st.session_state.user_input:
    sentiment = text_clf.predict([st.session_state.user_input])
    sentiment_label = sentiment[0]

    sentiment_styles = {
        'positive': {'color': 'white', 'background_color': 'green'},
        'negative': {'color': 'white', 'background_color': 'red'},
        'neutral': {'color': 'black', 'background_color': 'yellow'}
    }

    style = sentiment_styles.get(sentiment_label, {'color': 'black', 'background_color': 'gray'}) #default colors if sentiment_label is not found

    st.markdown(
        f'<p style="background-color: {style["background_color"]}; color: {style["color"]}; padding: 10px;">'
        f'The sentence was evaluated as {sentiment_label}.</p>',
        unsafe_allow_html=True
    )