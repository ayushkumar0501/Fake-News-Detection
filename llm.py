import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

def explain_news(news, prediction, articles):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are an advanced real-time fact-checking AI.

User Claim:
{news}

Dataset-Based ML Prediction:
{prediction}

Recent News Evidence:
{articles}

Instructions:

1. Use recent news evidence as highest priority.
2. If trusted sources directly support claim → mark REAL
3. If claim is exaggerated → mark PARTIALLY REAL / MISLEADING
4. If claim is clearly false → mark FAKE
5. Ignore outdated internal knowledge if recent evidence says otherwise.
6. Deeply compare article descriptions before giving final answer.

Output format:

Final Fact Check Result: [REAL / PARTIALLY REAL / FAKE]

Reason:
Clear explanation.
"""

    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    result = response.json()

    print("\nAI Explanation (Generated using OpenRouter LLM + NewsAPI Context):")
    print(result["choices"][0]["message"]["content"])