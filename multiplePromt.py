import os 
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain #Chain to run queries against LLMs.
from langchain.chains import SimpleSequentialChain # instead of calling propmt 1 n 2 separately I can combine both 
# and call them sequentially using this library

#to keep the memory of chat
from langchain.memory import ConversationBufferMemory

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

#memory 

person_memory= ConversationBufferMemory(input_key="quote", memory_key= "chat_history")
dob_memory= ConversationBufferMemory(input_key="title", memory_key= "chat_history")
descr_memory= ConversationBufferMemory(input_key="dob", memory_key= "description_history")


#OpenAI LLM model initialisation 
llm= OpenAI(temperature= 0.8)

## chain Initialisation create a var called chain and initialise chain with all params
chain= LLMChain(llm= llm, prompt=first_input_prompt, verbose= True, output_key= 'person', memory=person_memory) 
# when there is multiple prompt, the input I have given Andy and that should be the output of that prompt as well
# so that the conversation about Andy Samberg can be continued with multiple prompts

#SECOND PROMPT
second_input_prompt= PromptTemplate(
    input_variables= ['person'], #here input will be o/p of 1st prompt.
    template= "When was {person} born?" #the placeholder considered here is the output of 1st prompt that is person
)
# here the output for when he was born will be the DOB, that will be considered as input for next prompt

#second prompt chain
chain2= LLMChain(llm= llm, prompt=second_input_prompt, verbose= True, output_key= 'dob', memory= dob_memory) 


#third prompt
third_input_prompt= PromptTemplate(
    input_variables= ['dob'], 
    template= "Mention 5 major event happened around {dob}"
#second prompt's output key is dob, so taking that I am wrtiting thrid prompt and asking question around it
)
#third prompt chain
chain3= LLMChain(llm= llm, prompt=third_input_prompt, verbose= True, output_key= 'description', memory= descr_memory) 

#instead of running each prompt separately, I wi ll combine both in sequence. Create a new var and call SimpleSequentialChain

# Parent_Chain= SimpleSequentialChain(chains= [chain, chain2], verbose=True)

Parent_Chain= SimpleSequentialChain(
    chains= [chain, chain2, chain3], input_variables=['name'], output_variables=['person', 'dob','description'],verbose=True)

if input_text:
    st.write(Parent_Chain.run({'name': input_text}))

    with st.expander("Person Name"):
        st.info(person_memory.buffer)

    with st.expander("Major Events"):
        st.info(descr_memory.buffer)
