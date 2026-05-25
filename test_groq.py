import os
import requests

api_key = os.environ.get("GROQ_API_KEY", "VOTRE_CLE_API")
headers = {"Authorization": f"Bearer {api_key}"}

try:
    response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
    if response.status_code == 200:
        models = response.json().get("data", [])
        print("Available Groq Models:")
        for m in models:
            print(f"- {m['id']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Exception: {e}")
