from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
import os

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, google_api_key=os.getenv("GEMINI_KEY"))

prompt= PromptTemplate.from_template(
    "Explain in 5 lines about {topic}"
)

parser= StrOutputParser()

chain=  prompt | llm | parser

result= chain.invoke({"topic" : "Python Programming Language"})
print(result)


# to check the inner working of chain the steps 

chain.get_graph().print_ascii()

