## Integrate our code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
# from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title('Celebrity Search Results')
input_text = st.text_input("Search the topic u want")

# Prompt Templates
first_prompt_input = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about celebrity {name}"
    
)
person_memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='chat_history')
desc_meory =  ConversationBufferMemory(input_key='dob',memory_key='chat_history')

## OPENAI LLMS

llm = OpenAI(temperature=0.8) 
chain = LLMChain(llm=llm, prompt = first_prompt_input,verbose=True,output_key="person",memory=person_memory)

# This is for only we have single prompt
# if input_text:
#     st.write(chain.run(input_text))

# Prompt Templates
second_prompt_input = PromptTemplate(
    input_variables = ['person'],
    template = "When was {person} born"
    
)

second_chain = LLMChain(llm=llm, prompt = second_prompt_input,verbose=True,output_key="dob",memory=dob_memory)

# Prompt Templates
third_prompt_input = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention 5 major events happend around {dob} in the world"
    
)

third_chain = LLMChain(llm=llm, prompt = third_prompt_input,verbose=True,output_key="description",memory=desc_meory)


# parent_chain=SimpleSequentialChain(chains=[chain,second_chain],verbose=True)
parent_chain=SequentialChain(chains=[chain,second_chain,third_chain],input_variables=['name'],
                             output_variables=['person','dob','description'],verbose=True)

# if input_text:
#     st.write(parent_chain.run(input_text))

if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)
        
    with st.expander('Major Events'):
        st.info(desc_meory.buffer)