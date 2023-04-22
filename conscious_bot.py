import streamlit as st
from streamlit_chat import message
import os
import openai_util


st.set_page_config(page_title="Consciousness-QA-bot", page_icon=":robot:")
st.header("Ask your Question about conciousness ")
st.markdown("##### based on 'Heartfullness-Dec17 Magazine - What is Consciousness' ?")
st.divider()

##### Get Open AI API Key from User
def get_api_key():
    st.markdown("Provide your OpenAI API Key")
    input_text = st.text_input(label="check out this to get OpenAI API Key. \
        [link](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)", 
        placeholder="Ex: sk-2twmA8tfCb8un4...",
        key="openai_api_key_input")
    st.write()

    return input_text

openai_api_key = get_api_key() 

# model = st.selectbox(
#     "Select a model",
#     ("gpt-3.5-turbo (default)", "gpt-4 (make sure you have access to this model)","text-davinci-003","text-ada-001")
# )

## pinecone init
index_name = "consciousness-openai"
pincone_index = openai_util.init_pinecone(index_name)
#chain, pineconeDocSearchHandler = openai_util.init_openAI_embedding_from_pinecone(pincone_index)


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
    

query = st.text_input("Query: ", key="input")

if 'messages' not in st.session_state:
    st.session_state['messages'] = openai_util.get_initial_message()
    

if query:
    with st.spinner("generating..."):
        messages = st.session_state['messages']
        messages = openai_util.update_chat(messages, "user", query)
        response = openai_util.openAI_get_response(pincone_index,query,openai_api_key)
        
        #response = openai_util.get_chatgpt_response(messages,model)
        messages = openai_util.update_chat(messages, "assistant", response)
        st.session_state.past.append(query)
        st.session_state.generated.append(response)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

# with st.expander("Show Messages"):
#     st.write(st.session_state['messages'])
    