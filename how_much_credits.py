import requests
import os
from dotenv import load_dotenv

load_dotenv()
# Set your OpenRouter API Key
api_key = os.getenv("OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/credits"

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json().get("data", {})
    total_credits = data.get("total_credits", 0)
    total_usage = data.get("total_usage", 0)
    remaining_balance = total_credits - total_usage
    
    print(f"Total Credits: ${total_credits:.4f}")
    print(f"Total Usage: ${total_usage:.4f}")
    print(f"Remaining Balance: ${remaining_balance:.4f}")
else:
    print(f"Error: {response.status_code} - {response.text}")


