import os
import re
import openai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Test keywords.
keywords = ["hotel", "contract", "wife", "category", "dealer"]

openai.api_key = os.getenv("OPENAI_API_KEY")
prompt = f"Generate a prompt for an LLM with these words: {', '.join(keywords)}"
response = openai.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    temperature=0.8,
    max_tokens=500
)
print(re.sub("\n", "", response.choices[0].text).strip("\""))