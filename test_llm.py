import requests

url = "http://localhost:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "Qwen/Qwen3.5-27B",
    "messages": [
        {"role": "user", "content": "What is 2+2? Answer in valid JSON like {'answer': 4}"}
    ],
    "extra_body": {"enable_thinking": False},
}

response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.text)
