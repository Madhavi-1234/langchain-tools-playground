from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from dotenv import load_dotenv   # type: ignore 
from langchain_core.prompts import PromptTemplate    # type: ignore  
from langchain_core.output_parsers import StrOutputParser  # type: ignore 
from langchain_core.runnables import RunnableParallel   # type: ignore 


load_dotenv()
import os

llm1= ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, google_api_key=os.getenv("GEMINI_KEY"),max_tokens=200)
llm2= ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=os.getenv("GEMINI_KEY"),max_tokens=100)


prompt1= PromptTemplate(
    template="Explain and write detailed notes  for the given {text}",
    input_variables=["text"]
)

prompt2= PromptTemplate(
    template="summarize in short questions about the text\n{text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
template="merge the provided notes and quiz into a single document \n {notes} \n {quiz}",
input_variables=["notes", "quiz"]
)

parser= StrOutputParser()

parallel_chain= RunnableParallel({
    'notes':prompt1 |llm1 | parser ,
    'quiz':prompt2 | llm2 |parser
})

merge_chain= prompt3 | llm1 | parser

chain= parallel_chain | merge_chain

text="""
Artificial intelligence is one of the emerging and in-demand fields in computer science. The artificial intelligence syllabus focuses on providing you with a strong understanding of AI principles and tools and how they can be applied to solve real-world problems. Artificial Intelligence courses are offered at various levels with BTech courses being one of the most popular choices.

In the artificial intelligence syllabus, you will learn about topics like data structures, programming, algorithms, machine learning, deep learning, neural networks and more. The curriculum is also updated regularly to include the latest advancements. The syllabus generally consists of core and elective subjects along with practical learning through laboratories and internships.
"""


result= chain.invoke({"text": text})
print(result)


# to get the inner working of chain the steps

chain.get_graph().print_ascii()