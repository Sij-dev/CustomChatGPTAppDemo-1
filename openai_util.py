import openai
from langchain.vectorstores import  Pinecone
from tqdm.autonotebook import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

import streamlit as st

from langchain.chains.question_answering import load_qa_chain

import pinecone
import os

#### uncomment for running locally #######
# from dotenv import load_dotenv
# if load_dotenv():
#     pinecone_api_key = os.getenv('PINECONE_API_KEY')
#     pinecone_api_env = os.getenv('PINECONE_API_ENV')  
    #OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  
    
#for streamlit cloud - comment to run locally 
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_api_env = st.secrets["PINECONE_API_ENV"]

def init_pinecone(idx_name = "consciousness-openai"):
    #pinecone index name
    index_name = idx_name

    # initialize pinecone
    pinecone.init(
        api_key=pinecone_api_key,  # find at app.pinecone.io
        environment=pinecone_api_env  # next to api key in console
    )
    
    index = pinecone.Index(index_name)
    return index

    
def openAI_get_response(index ,message, openai_api_key,model = "gpt-3.5-turbo" ):
    
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) #model = model
    except ValueError:
        st.error("Please enter a valid Open API")
        

    docsearch = Pinecone(index, embeddings.embed_query, 'text')
    docs = docsearch.similarity_search(message, include_metadata=True)
    
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key) #model_name= model
    chain = load_qa_chain(llm, chain_type="stuff")

    docs = docsearch.similarity_search(query=message)
    response = chain.run(input_documents=docs, question=message)
    return response


def get_initial_message():
    
    messages=[
            {"role": "system", "content": " You are a spiritual person and heartfullness practioner.\
              Hearfullness is the meditation technique based on yogic transmission., You will explain \
             spirtiual related question in simple human understandable format with examples.\
             Also you will explain differnt perspectives of the question and answer accordingly. \
             You will focus on the question and input information  provided along with the question \
             to provide the response."},
        
        ]
    return messages

def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages

def get_chatgpt_response(messages, model='gpt-3.5-turbo'):
    print("model: ", model)
    
    # response = openai.ChatCompletion.create(
    # model=model,
    # messages=messages, 
    # temperature = 0  
    # )
    # return  response['choices'][0]['message']['content']



