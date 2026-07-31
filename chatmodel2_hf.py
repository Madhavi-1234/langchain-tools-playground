from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HUGGING_FACE_HUB_ACCESS_TOKEN")
print(token)  # Optional, to check if it loads correctly

llm= HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation")

model= ChatHuggingFace(llm=llm)


print("Hugging Face chatbot (type 'exit' to quit chat)\n")


while True:
    try:
        user_input = input("You: ")
    except KeyboardInterrupt:
        print("\nGoodbye")
        break

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye")
        break

    response = model.invoke(user_input)
    print("Hugging Face:", response.content, "\n")
