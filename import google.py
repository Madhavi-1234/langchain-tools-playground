import google.generativeai as genai

genai.configure(api_key="AIzaSyBuKZJ1-oLfkFE2lqlFHoFuv-SROZBwmrs")

models = genai.list_models()
for m in models["models"]:
    print(m["name"], "-", m.get("description", ""))
