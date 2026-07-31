from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
import os
load_dotenv()


llm= HuggingFacePipeline.from_model_id( 
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation",
    pipelne_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)


model = ChatHuggingFace(llm=llm)

while True:
    user_input= input("You:")
    if user_input.lower=="exit":
        print("Goodbye!")
        break
    response= model.invoke(user_input)
    print("HuggingFace:",response.content)
        