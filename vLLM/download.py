import requests
token = "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"
headers = {"Authorization": f"Bearer {token}"}
response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
print(response.json())