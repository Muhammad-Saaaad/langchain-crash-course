import os

from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")

print(key)
print(type(key))