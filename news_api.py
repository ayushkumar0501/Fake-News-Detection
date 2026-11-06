import requests
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get NewsAPI key
api_key = os.getenv("NEWS_API_KEY")


# Extract useful keywords from user query
def extract_keywords(query):
    stopwords = {
        "is", "the", "of", "a", "an",
        "in", "to", "for", "on", "and",
        "had", "has", "have", "won",
        "was", "were", "will", "be",
        "did", "does", "do", "joined"
    }

    # Convert query into lowercase words
    words = re.findall(r'\w+', query.lower())

    # Remove stopwords
    keywords = [word for word in words if word not in stopwords]

    # Return first 4 useful words
    return " ".join(keywords[:4])


# Fetch news articles
def fetch_news(query):
    keywords = extract_keywords(query)

    print(f"\nSearching Live NewsAPI Articles Using Keywords: {keywords}")

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={keywords}&"
        f"sortBy=publishedAt&"
        f"language=en&"
        f"apiKey={api_key}"
    )

    response = requests.get(url)

    print("\nNewsAPI Status Code:", response.status_code)

    data = response.json()

    articles = data.get("articles", [])

    article_data = []

    if articles:
        print("\nRelated Articles Found (From NewsAPI Real-Time Data):\n")

        for article in articles[:3]:
            title = article.get("title", "No Title")
            description = article.get("description", "No Description")
            source = article.get("source", {}).get("name", "Unknown Source")
            url_link = article.get("url", "No URL")

            print(f"Title: {title}")
            print(f"Source: {source}")
            print(f"Description: {description}")
            print(f"URL: {url_link}")
            print("-" * 60)

            article_data.append({
                "title": title,
                "description": description,
                "source": source,
                "url": url_link
            })

    else:
        print("\nNo related trusted news found.")

    return article_data