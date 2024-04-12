import os 
import streamlit as st
import pickle
import langchain
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
import faiss as FA
from langchain.agents import AgentType,load_tools,initialize_agent
from langchain.memory import ConversationBufferWindowMemory

os.environ['OPENAI_API_KEY'] = ''
os.environ['SERPAPI_API_KEY'] = ''
model = OpenAI(temperature=0.7)
tools = load_tools(['serpapi'],llm=model)
st.title("Chat News")
st.sidebar.title("New's Articles URL")
urls_list = []
for i in range(3):
    url = st.sidebar.text_input(f"URL: {i+1}")
    urls_list.append(url)
process_clicked = st.sidebar.button("Process URL")

st.empty()
file_path = 'vector_store.pkl'

memory = ConversationBufferWindowMemory(k=3)
google = st.sidebar.text_input("Enter a google search Prompt")
clicked = st.sidebar.button("Process")
agent = initialize_agent(
    tools,model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=False,memory= memory
)


if process_clicked:
    loader = UnstructuredURLLoader(urls = urls_list)
    data = loader.load()
    st.text("Data Loading Started --- -- -- --- -- --- -")
    splitter  = RecursiveCharacterTextSplitter(separators=['\n\n','\n','-'],chunk_size = 2000)
    docs = splitter.split_documents(data)


    embedding = OpenAIEmbeddings()
    vector = faiss.FAISS.from_documents(docs,embedding)
    st.text("Embedding vectors....vectors started building --- - --------- -- -- - ---- ------ --------- ---- -")

    with open(file_path,'wb') as file:
        pickle.dump(vector.index_to_docstore_id,file)

query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as file:
            vectors = pickle.load(file)
        
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm = model,
            retriever = vector.as_retriever()
        )
        result = chain({'question': query})
        langchain.debug = True
        st.header("Here is my Answer")
        st.subheader(result['answer'])
        st.header("Sources:")
        st.subheader(result['sources'])
    
if clicked:
    result = agent.run(google)
    st.header("Google Search Done:")
    st.subheader(result)
