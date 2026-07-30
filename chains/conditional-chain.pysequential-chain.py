from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
import os
llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, google_api_key=os.getenv("GEMINI_KEY"))


prompt1= PromptTemplate(
    template="Explain in detailed 5 lines about {topic}",
    input_variables=["topic"]
)
prompt2= PromptTemplate(
    template="summarize in 3 key points about the text\n{text}",
    input_variables=["text"]
)
parser= StrOutputParser()

# Chain 1: Topic -> Explanation
chain1= prompt1 | llm | parser

# Chain 2: Explanation -> Summary
chain2= prompt2 | llm | parser

# Execute chain 1
result1 = chain1.invoke({"topic" : "Unemployment"})
print("Explanation:")
print(result1)
print("\n" + "="*50 + "\n")

# Execute chain 2
result2 = chain2.invoke({"text" : result1})
print("Summary:")
print(result2)

# to check the inner working of chain the steps


