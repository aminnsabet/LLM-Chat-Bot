from openai import OpenAI

# init client and connect to localhost server
client = OpenAI(
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0IiwiZXhwIjoxNzE5NDg4NjM4fQ.UeilH0thju-EZNfSKSSalYF1N03SSCLFWMqZyn15Mcg",
    base_url="http://localhost:8086" # change the default port if needed
)

# call API
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-1337-turbo-pro-max",
)

# print the top "choice" 
print(chat_completion.choices[0].message.content)