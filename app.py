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

result = text_clf.predict(['Hate it'])
st.write(result)

st.write("""
# Sentiment Analysis
Write a sentance to evaluate.
""")