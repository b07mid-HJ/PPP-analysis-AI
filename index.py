import os
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_vector import MultiVectorRetriever
import streamlit as st
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__fe037fb4637146b8bdcf8355ea7c7b79"
os.environ["LANGCHAIN_PROJECT"] = "Jade-chatbot"

if "history" not in st.session_state:
    st.session_state.history = []

#model preperation
os.environ["GOOGLE_API_KEY"]="AIzaSyADkFFYD1o6HCPgZ7ftisu8c0Dv40SWQ80"
DATABASE_URL = "postgresql+psycopg2://postgres:admin@localhost:5432/Jade-Chatbot"

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs,
)

model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.5, 
                                convert_system_message_to_human=True
                            )
# from langchain_community.chat_models import ChatOllama
# model=ChatOllama(model="mistral:7b-instruct-v0.2-q5_1")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# The vectorstore to use to index the child chunks
vectorstore1 =PGVector(
    collection_name="table_summaries",
    connection_string=DATABASE_URL,
    embedding_function=hf,
)


with open(f"./chunks/store1.pkl", 'rb') as f:
    store1 = pickle.load(f)
# The storage layer for the parent documents
id_key1 = "doc_id"

# The retriever (empty to start)
retriever1 = vectorstore1.as_retriever()

vectorstore2 = PGVector(
    collection_name="child_chunks",
    connection_string=DATABASE_URL,
    embedding_function=hf,
)

with open(f"./chunks/store2.pkl", 'rb') as f:
    store2 = pickle.load(f)
id_key2 = "doc_id"
# The retriever (empty to start)
retriever2 = MultiVectorRetriever(
    vectorstore=vectorstore2,
    byte_store=store2,
    id_key=id_key2,
)


from langchain_core.runnables import RunnablePassthrough

# Prompt template
template = """
You are a Private-Public Partnership (PPP) feasibility expert. You are tasked with answering questions the feasibility of a PPP project.\n
Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
from langchain.retrievers import EnsembleRetriever


# LLMChainFilter
# from langchain.retrievers.document_compressors import LLMChainFilter
# compressor = LLMChainFilter.from_llm(model)

from langchain.retrievers.document_compressors import LLMChainExtractor
compressor = LLMChainExtractor.from_llm(model)

from langchain.retrievers import ContextualCompressionRetriever
ensemble=EnsembleRetriever(retrievers=[retriever1,retriever2],weights=[0.5,0.5])
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=ensemble
# )

# RAG pipeline
chain = (
    {"context": ensemble, "question": RunnablePassthrough()}
    | prompt
    | model
    |StrOutputParser()
)


st.title("Welcome to Jade Private Chat Assistant")


for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

prompt = st.chat_input("Say something")
if prompt:
    st.session_state.history.append({
        'role':'user',
        'content':prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('💡Thinking'):
        response = chain.invoke(prompt)

        st.session_state.history.append({
            'role' : 'Assistant',
            'content' : response
        })

        with st.chat_message("Assistant"):
            st.markdown(response)