from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY') 

# Initialize Claude through LangChain
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

#Generate a response
response = llm.invoke("What is organizational AI readiness?")
print(response.content)

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# llm2=ChatOpenAI(model='gpt-4o',temperature=0)
# response_op = llm2.invoke("Hi")
# print(response_op.content)