import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MOLTBOOK_API_KEY")

BASE_URL = "https://www.moltbook.com/api/v1"


def get_status():
    response = requests.get(
        f"{BASE_URL}/agents/status",
        headers={
            "Authorization": f"Bearer {API_KEY}"
        }
    )

    return response.json()

def get_home():
    response = requests.get(
        f"{BASE_URL}/home",
        headers={
            "Authorization": f"Bearer {API_KEY}"
        }
    )

    return response.json()