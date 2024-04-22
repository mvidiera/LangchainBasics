#main.py has the skeleton, copying and pasting here
import os 
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain #Chain to run queries against LLMs.

import streamlit as st

os.environ['OPENAI_API_KEY']= openai_key 


#intitializing streamlit framework

st.title("Celebrity search results")

input_text= st.text_input("Search any topic")

# Customised own template: propmt templates

first_input_prompt= PromptTemplate(
    input_variables= ['name'],
    template= "tell me about celebrity {name}" #placeholder
)

#OpenAI LLM model initialisation 
llm= OpenAI(temperature= 0.8)

## chain Initialisation create a var called chain and initialise chain with all params
chain= LLMChain(llm= llm, prompt=first_input_prompt, verbose= True ) 

#params: llm-> model, prompt-> what is the prompt, verbose-> 
#this callback support (chains, models, agents, tools, retrievers) to print the inputs they receive and outputs they generate. 
#in short: asking pgm to tell everything what it is doing all the time. like logging


if input_text:
    st.write(chain.run(input_text))

#control-> whatever I enter in streamlt search bar, example: input_text=Andy Samberg. control goes to chain. 
#using llm model, it considers template like "tell me about celeb ANdy Samberg"
# command/search will be executed about Andy, then result will be returned back to streamlit and write() command is used to show the result. 


