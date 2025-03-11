# Sentiment Analysis

## Table of contents
- [About](#about)
    - [Key features](#key-features)
    - [Data](#data)
- [Getting started](#getting-started)
     - [Dependencies](#dependencies)
     - [Dev setup step-by-step](#dev-setup-step-by-step)


## About

This project is an exercise in developing and deploying a mobile-friendly Sentiment Analysis frontend. It was created as part of the Cloud Computing course at Lapland University of Applied Sciences. The project is built with Python and deployed using Streamlit. You can find it running live on Streamlit: https://iinaus-sentiment-analysis.streamlit.app/

The application performs sentiment analysis on user input text, classifying it into positive, negative, or neutral categories. The model is trained using a machine learning pipeline, which includes a TfidfVectorizer for transforming text into numerical features and a LinearSVC classifier for sentiment prediction.

With this exercise we practiced:
- Developing and deploying a machine learning-based web application using Streamlit
- Handling text data and implementing machine learning workflows
- Enhancing the user experience with interactive UI elements

### Key Features
- **User Input**: Users can type a sentence into a text box, and the app will predict its sentiment.
- **Real-Time Feedback**: The app dynamically displays the predicted sentiment along with an appropriate color-coded message.
- **Mobile-Friendly**: The application is designed to be responsive and user-friendly on mobile devices.

### Data

The model is trained using a CSV dataset (train.csv) that contains labeled text data. The dataset consists of text samples labeled with sentiments (positive, negative, neutral). The dataset used in this project is based on the [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset) from Kaggle, which is a collection of tweets and their sentiment labels. You can find more details and download the dataset from the provided link.

## Getting started

Follow the instructions below to set up the project and run the app locally.

### Dependencies

This project relies on the following dependencies:

- **Python**: The main programming language used for the backend and model training.
- **Streamlit**: A Python library used to quickly build and deploy the frontend.
- **Scikit-Learn**: Used for machine learning, including the TfidfVectorizer and LinearSVC classifier.
- **Pandas**: Used for handling and preprocessing the dataset.

Ensure these dependencies are installed to ensure smooth execution of the project.

### Dev setup step-by-step

1. Clone the project
2. Install the required dependencies with the following command
 `pip install -r requirements.txt`
3. Run the Streamlit app with the following command
`streamlit run app.py`
4. Enter a sentence in the input box and the app will display the predicted sentiment