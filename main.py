import pickle
from news_api import fetch_news
from llm import explain_news

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

news = input("Enter News Headline/Text: ")

# ML Prediction
news_vector = vectorizer.transform([news])
prediction = model.predict(news_vector)

ml_result = "REAL" if prediction[0] == 1 else "FAKE"

print(f"\nDataset-Based Prediction (Trained on Fake.csv + True.csv): {ml_result}")

# Fetch articles
articles = fetch_news(news)

# LLM explanation
explain_news(news, ml_result, articles)