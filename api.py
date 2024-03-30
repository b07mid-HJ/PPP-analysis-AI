import datetime
from operator import itemgetter
import os
import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from contextlib import asynccontextmanager
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-e3eed82a-f92e-4f2b-87e1-475b384a0754"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-fc9c786d-33c2-4c04-80ae-365411b1e9c9"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

DATABASE_URL = "postgresql+psycopg2://postgres:admin@localhost:5432/Jade-Chatbot"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "chatusers"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    def verify_password(self, password: str):
        return bcrypt.checkpw(
            password.encode("utf-8"), self.hashed_password.encode("utf-8")
        )


class UserIn(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    username: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


os.environ["GOOGLE_API_KEY"]="AIzaSyADkFFYD1o6HCPgZ7ftisu8c0Dv40SWQ80"

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(lifespan=lifespan)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
SECRET_KEY = "your_jwt_secret_key"


def authenticate_user(username: str, password: str, db: Session):
    user = db.query(User).filter(User.username == username).first()
    if not user or not bcrypt.checkpw(
        password.encode("utf-8"), user.hashed_password.encode("utf-8")
    ):
        return False
    return user


def create_access_token(data: dict, expires_delta: datetime.timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


def get_langfuse():
    return Langfuse()


def get_trace_handler(
    langfuse: Langfuse = Depends(get_langfuse), user=Depends(get_current_user)
):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated"
        )
    trace = langfuse.trace(user_id=user.username)
    return trace.get_langchain_handler()


@app.post("/register", response_model=UserOut)
def register(user_in: UserIn, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user_in.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = bcrypt.hashpw(user_in.password.encode("utf-8"), bcrypt.gensalt())
    new_user = User(
        username=user_in.username, hashed_password=hashed_password.decode("utf-8")
    )
    db.add(new_user)
    db.commit()
    return UserOut(username=new_user.username)


@app.post("/token")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/chat/")
async def quick_response(
    question: str, user=Depends(get_current_user), handler=Depends(get_trace_handler)
):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated"
        )
    query = question
    result = await chain.ainvoke(query, config={"callbacks": [handler]})
    return result