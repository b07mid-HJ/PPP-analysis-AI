import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import db
import jwt
import extra_streamlit_components as stx
from langfuse.callback import CallbackHandler
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
from langfuse import Langfuse
from streamlit_extras.stylable_container import stylable_container
from streamlit_option_menu import option_menu


# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Jade Software", page_icon=":bar_chart:", layout="wide")

cookie_manager = stx.CookieManager()
# --- DEMO PURPOSE ONLY --- #
placeholder = st.empty()
# ------------------------- #

# --- USER AUTHENTICATION ---
users = db.fetch_all_users()

# Initialize an empty dictionary for usernames
usernames_dict = {}
# Populate the dictionary with usernames and their hashed passwords
for user in users:
    # Assuming user[1] is the username and user[2] is the hashed password
    username = user[1]
    hashed_password = user[2]
    name=user[3]
    
    # Populate the usernames dictionary
    usernames_dict[username] = {'name':name, 'password': hashed_password}

# Now, structure the credentials dictionary as expected by streamlit_authenticator
credentials = {
    'usernames': 
        usernames_dict
    
}

authenticator = stauth.Authenticate(credentials, "Jade6co", "benchmark-key-2244",cookie_manager)

name, authentication_status, username = authenticator.login()

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    logo = "./assets/jade_advisory_logo-removebg-preview.png"
    st.sidebar.image(logo, width=80)
    authenticator.logout("Logout", "sidebar")
    def chat():
        def get_current_user():
            value = cookie_manager.get(cookie="Jade6co")
            if value is not None:
                payload = jwt.decode(value, "benchmark-key-2244", algorithms=["HS256"])
                return dict(payload)["username"]
            else:
                return None

        langfuse_handler = CallbackHandler(
            secret_key="sk-lf-e3eed82a-f92e-4f2b-87e1-475b384a0754",
            public_key="pk-lf-fc9c786d-33c2-4c04-80ae-365411b1e9c9",
            host="http://localhost:3000",
            user_id=get_current_user())


        #model preperation
        os.environ["GOOGLE_API_KEY"]="AIzaSyADkFFYD1o6HCPgZ7ftisu8c0Dv40SWQ80"
        DATABASE_URL = "postgresql+psycopg2://postgres:admin@localhost:5432/Jade-Chatbot"
        @st.cache_resource
        def getembeddings():
            model_name = "BAAI/bge-m3"
            model_kwargs = {"device": "cpu"}
            hf = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs,
            )
            return hf
        hf=getembeddings()
        def list_doc():
            l=db.fetch_all_docments()
            ll=[i[0] for i in l]
            return ll
        listdoc=list_doc()
        
        st.title("Welcome to Jade Private Chat Assistant")
        cols = st.columns([2,1])

        option=cols[0].selectbox("Select a model", listdoc)
        if "history" not in st.session_state:
            st.session_state.history = db.fetch_conversations_by_user_document(get_current_user(),option)
        st.session_state.history = db.fetch_conversations_by_user_document(get_current_user(),option)
        cols[1].markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
        def clear_hist(user,doc):
            db.clear_chathistory(user,doc)
        if cols[1].button("Clear Chat History"):
            clear_hist(get_current_user(),option)
        st.session_state.history = db.fetch_conversations_by_user_document(get_current_user(),option)
        st.write("##")
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
            collection_name=f"{option}: table_summaries",
            connection_string=DATABASE_URL,
            embedding_function=hf,
        )


        with open(f"./chunks/{option}/store1.pkl", 'rb') as f:
            store1 = pickle.load(f)
        # The storage layer for the parent documents
        id_key1 = "doc_id"

        # The retriever (empty to start)
        retriever1 = vectorstore1.as_retriever()

        vectorstore2 = PGVector(
            collection_name=f"{option}: child_chunks",
            connection_string=DATABASE_URL,
            embedding_function=hf,
        )

        with open(f"./chunks/{option}/store2.pkl", 'rb') as f:
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

        template1 = """
        Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        You need to keep on consideration the context of the conversation and the question \
        just reformulate to adapte to the context of the conversation it if needed and otherwise return it as is,\
        **ALWAYS RETURN THE QUESTION IN ENGLISH**.\n
        here is the conversation:
        {convo}
        Question: {question}
        """
        prompt1 = ChatPromptTemplate.from_template(template1)
        prompt1 = prompt1.partial(convo=str(st.session_state.history))
        chain1 = (
            {"question": RunnablePassthrough()}
            | prompt1
            | model
            | StrOutputParser()
        )
        
        
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
                q2=chain1.invoke(prompt,config={"callbacks": [langfuse_handler]})
                response = chain.invoke(q2,config={"callbacks": [langfuse_handler]})

                st.session_state.history.append({
                    'role' : 'Assistant',
                    'content' : response
                })

                with st.chat_message("Assistant"):
                    st.markdown(response)
            db.insert_or_update_conversation(get_current_user(), st.session_state.history,option)
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",  # required
            options=["chat", "gene"],  # required
            icons=["house", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
        )
    if selected == "chat":
        chat()