from mistralai.client import Mistral
import os


with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", "h2kaR0bMtO27DzCWXBy0JU0cUhnGMtUw"),
) as mistral:

    res = mistral.chat.complete(model="mistral-large-latest", messages=[
        {
            "role": "user",
            "content": "Who is the best French painter? Answer in one short sentence.",
        },
    ], stream=False, response_format={
        "type": "text",
    })

    # Handle response
    print(res)


