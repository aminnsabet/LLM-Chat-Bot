from openai import OpenAI

client = OpenAI(api_key="fake-api-key", base_url="http://localhost:8086/v1/")

response = client.chat.completions.create(
    model="TheBloke/Llama-2-7B-Chat-AWQ",
    messages=[{"role": "user", "content": "HI"}],
    max_tokens=20,
    temperature=0.1,
)

print(response)
