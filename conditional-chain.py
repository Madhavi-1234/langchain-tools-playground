from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal



load_dotenv()
import os

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, google_api_key=os.getenv("GEMINI_KEY"),max_tokens=200)
 
parser= StrOutputParser


class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative']= Field(description= "sentiment of the feedback either positive or negative")

parser1= PydanticOutputParser(pydantic_object=Feedback)


prompt1= PromptTemplate(
    template="classfiy the given feeback into positive or negative \n {text} \n {format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser1.get_format_instructions()}
)

classifier_chain= prompt1 | llm | parser

prompt2= PromptTemplate(
    template=" wriite an appropriate feedback to this postive feedback \n {text}",
    input_variables=["text"]
)


prompt3= PromptTemplate(
    template=" write an appropriate feedback to this negative feedback \n {text}",
    input_variables=["text"]
)


branch_chain= RunnableBranch(
    lambda x : x.sentiment=="positive",  prompt2 | llm | parser,
    lambda x : x.sentiment=="negative", prompt3 | llm | parser,
    RunnableLambda(lambda x: "not able to classify the feedback"), parser
    


)



chain= classifier_chain | branch_chain

result= chain.invoke({"Feedback": "This is the terrible smartphone"})




print(result)

chain.get_graph().print_ascii()




