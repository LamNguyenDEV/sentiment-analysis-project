# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
import nltk



# Download stopwords (first-time usage)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load the Rotten Tomatoes dataset
data = pd.read_csv('train.csv')

# Check the first few rows of the dataset
print(data.head())
# Data Preprocessing

# Function to clean text (removing special characters, converting to lowercase, etc.)
def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

 # Apply cleaning function to the reviews
data['cleaned_review'] = data['text'].apply(clean_text)   

# Tokenization, stopword removal, and lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# # Apply preprocessing
data['processed_review'] = data['cleaned_review'].apply(preprocess_text)

# Split data into features (X) and labels (y)
X = data['processed_review']
y = data['label']  # 'positive' or 'negative'

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model Training using Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Implement model for given text review 
# Function to preprocess the new review
def preprocess_new_review(text):
    # Clean the review (remove special characters, lowercase)
    text = clean_text(text)
    # Tokenize, remove stopwords, and lemmatize the words
    text = preprocess_text(text)
    return text

# Let's assume we have a new review:
new_review = "The movie was absolutely amazing! I loved every moment of it, especially the acting."

# Preprocess the new review
processed_review = preprocess_new_review(new_review)

# Convert the processed review into TF-IDF features
processed_review_tfidf = tfidf.transform([processed_review])

# Make prediction using the trained Naive Bayes model
predicted_sentiment = model.predict(processed_review_tfidf)

# Output the predicted sentiment
print(f"Review: {new_review}")
print(f"Predicted Sentiment: {predicted_sentiment[0]}")
