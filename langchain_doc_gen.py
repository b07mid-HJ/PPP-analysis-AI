import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["GOOGLE_API_KEY"]="AIzaSyADkFFYD1o6HCPgZ7ftisu8c0Dv40SWQ80"

model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.5, 
                                convert_system_message_to_human=True
                            )
# from langchain_community.chat_models import ChatOllama
# model=ChatOllama(model="mistral:7b-instruct-v0.2-q5_1")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def table_desc(title,table):
    prompt_text = """You are an assistant tasked with table analysis.\ 
    Please write a concise description of the table, highlighting the most important aspects of the data.\
    ** Focus on the key trends, comparisons, or insights revealed by the table.\
    ** Use clear and concise language, avoiding repetition of information from the table itself.\
    ** If applicable, mention any significant values or relationships within the data.\
    ** Aim for a length of 2-3 sentences.\
    *THE RESPONSE SHOULD ONLY BE IN ENGLISH* \
    The title of this section is :{title}
    Give me a description of the table. Table : {table} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    summarize_chain = {"title": RunnablePassthrough(),"table":RunnablePassthrough()} | prompt | model | StrOutputParser()

    return summarize_chain.invoke({"title": title, "table": table})

def doc_content(title,content):
    prompt_text = """You are an assistant tasked with generating conclutions inside a document for a specific section.\ 
    Please write a concise conclution of the the section {title}, focusing on the key points and main ideas.\
    ** Based on the key findings from the descriptions (provided below), write a concluding summary for this section.\
    ** Focus on the most important insights and trends revealed by the data presented in the descriptions.\
    ** Briefly mention any significant findings or relationships highlighted in the descriptions.\
    ** Aim for a concise summary of 3-5 sentences that captures the overall takeaways from the section.\
    *THE RESPONSE SHOULD ONLY BE IN ENGLISH* \
    The title of this section is :{title}
    List of tables descriptions with their respective titles in this section : {content} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    summarize_chain = {"title": RunnablePassthrough(),"content":RunnablePassthrough()} | prompt | model | StrOutputParser()

    return summarize_chain.invoke({"title": title, "content": content})