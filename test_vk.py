import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("VK_TOKEN")
V = os.getenv("VK_API_VERSION", "5.131")

def vk_call(method, params=None):
    if params is None:
        params = {}

    params["access_token"] = TOKEN
    params["v"] = V

    url = f"https://api.vk.com/method/{method}"

    r = requests.get(url, params=params)
    return r.json()


print("Получаем список моих групп...")

data = vk_call("groups.get", {
    "extended": 1,
    "count": 1000
})

print(data)