from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_KEY")
print(f"API Key: {api_key[:20]}..." if api_key else "No API Key found")

genai.configure(api_key=api_key)

# List available models
print("\nAvailable models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  - {model.name}")
