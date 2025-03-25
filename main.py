import pickle
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.future import select
from sqlalchemy import Column, TEXT, LargeBinary, String
from sqlalchemy.sql import delete
from pydantic import BaseModel
import torch
import requests

#  Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device('cpu')
url = "https://api-inference.huggingface.co/models/google/gemma-3-27b-it"

#Hugging face Token(Create Yours and use it here)Limited usage
headers = {"Authorization": f"Bearer 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"}

# PostgreSQL Database Configuration
DATABASE_URL = "postgresql+asyncpg://postgres:Hi%40191003@localhost:5432/postgres"

async_engine = create_async_engine(DATABASE_URL, echo=True, future=True)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, expire_on_commit=False, class_=AsyncSession)

Base = declarative_base()

class EmbeddingModel(Base):
    __tablename__ = "embeddings"
    filename = Column(String(255), primary_key=True)
    embedding = Column(LargeBinary, nullable=False)  
    content = Column(TEXT, nullable=False)  # Adjusted for longer text storage

#  Create Tables
async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

#  Lifespan Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔹 Starting up... Creating tables if needed")
    await init_db()
    yield
    logger.info("🔹 Shutting down... Closing database connection")
    await async_engine.dispose()

#  FastAPI App with Lifespan
app = FastAPI(lifespan=lifespan)

#  Load Hugging Face Embedding Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#  Dependency for Database Session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

#  Upload Text File & Generate Embedding
@app.post("/uploadfile/")
async def upload_file(file: UploadFile, db: AsyncSession = Depends(get_db)):
    try:
        if not file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt files are allowed")

        content = await file.read()
        if not content.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        text_content = content.decode("utf-8")
        
         #  Check if the file already exists in the database
        existing_file = await db.execute(
            select(EmbeddingModel.filename).where(EmbeddingModel.filename == file.filename)
        )
        existing_file = existing_file.fetchone()  # Fetch the first row

        if existing_file:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' already exists. Please rename the file.")
        
        embedding_vector = model.encode(text_content)
        serialized_embedding = pickle.dumps(np.array(embedding_vector, dtype="float32"), protocol=pickle.HIGHEST_PROTOCOL)

        new_embedding = EmbeddingModel(filename=file.filename, embedding=serialized_embedding, content=text_content)
        db.add(new_embedding)

        await db.execute(delete(EmbeddingModel).where(EmbeddingModel.filename == 'AllEmbedding'))

        result = await db.execute(select(EmbeddingModel.embedding, EmbeddingModel.content).where(EmbeddingModel.filename != "AllEmbedding"))
        embeddings = []
        all_contents = ""

        for row in result.all():
            emb = pickle.loads(row[0])
            if isinstance(emb, np.ndarray):
                embeddings.append(emb)
            all_contents += row[1] + "\n"  # Combine all contents

        if embeddings:
            combined_embedding = np.mean(embeddings, axis=0)
            all_embedding = EmbeddingModel(
                filename="AllEmbedding", 
                embedding=pickle.dumps(combined_embedding, protocol=pickle.HIGHEST_PROTOCOL), 
                content=all_contents.strip()
            )
            db.add(all_embedding)

        await db.commit()
        return {"message": f" File '{file.filename}' processed and stored successfully"}
    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        logger.error(f" Error processing file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error. Could not process file.")

class TextBase(BaseModel):
    value: str

@app.post("/uploadText/")
async def upload_text(text1: TextBase, db: AsyncSession = Depends(get_db)):
    try:
        if not text1.value.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        embedding_vector = model.encode(text1.value)
        serialized_embedding = pickle.dumps(
            np.array(embedding_vector, dtype="float32"), protocol=pickle.HIGHEST_PROTOCOL
        )
        filename = f"text_{hash(text1.value)}.txt"
        new_embedding = EmbeddingModel(filename=filename, content=text1.value, embedding=serialized_embedding)
        db.add(new_embedding)

        await db.execute(delete(EmbeddingModel).where(EmbeddingModel.filename == 'AllEmbedding'))

        result = await db.execute(select(EmbeddingModel.embedding, EmbeddingModel.content).where(EmbeddingModel.filename != "AllEmbedding"))
        embeddings = []
        contents = []

        for row in result.all():
            emb = pickle.loads(row[0])
            if isinstance(emb, np.ndarray):
                embeddings.append(emb)
                contents.append(row[1])

        if embeddings:
            combined_embedding = np.mean(embeddings, axis=0)
            combined_content = "\n".join(contents)
            all_embedding = EmbeddingModel(
                filename="AllEmbedding",
                content=combined_content,
                embedding=pickle.dumps(combined_embedding, protocol=pickle.HIGHEST_PROTOCOL)
            )
            db.add(all_embedding)

        await db.commit()
        return {"message": " Text processed and stored successfully", "filename": filename}

    except Exception as e:
        logger.error(f" Error processing text: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error. Could not process embedding.")

class QuestionBase(BaseModel):
    filename: str
    question: str


#if the given filename exist retrive the data from it else retrieve data from AllEmbedding file(contain combination of all file data) 

#FileName passed should be of format fileName.txt
async def fetch_document_content(target_filename: str, db: AsyncSession) -> str:
    result = await db.execute(select(EmbeddingModel.content).where(EmbeddingModel.filename == target_filename))
    row = result.fetchone()
    return row[0] if row else None

@app.get("/retrieve/")
async def retrieve_relevant_text(question: QuestionBase, db: AsyncSession = Depends(get_db)):
    try:
        target_filename = question.filename

        # Check if the file exists in the database, else fallback to "AllEmbedding"
        existing_file = await db.execute(
            select(EmbeddingModel.filename).where(EmbeddingModel.filename == target_filename)
        )
        existing_file = existing_file.fetchone()

        if not existing_file:
            target_filename = "AllEmbedding"

        logger.info(f" Searching for content with filename: {target_filename}")
        logger.info(f" question: {question.question}")

        document_content = await fetch_document_content(target_filename, db)
        if not document_content:
            raise HTTPException(status_code=404, detail=f"No content found for '{target_filename}'")

        # Process large content with total text more than 5000
        response_text = process_large_content(document_content, question.question)

        return {"relevant_text": response_text}

    except Exception as e:
        logger.error(f" Error retrieving relevant text: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

def process_large_content(file_content: str, question: str) -> str:
    """Splits large text into chunks and recursively processes the content until it fits within limits."""
    max_chunk_size = 5000
    chunks = [file_content[i:i + max_chunk_size] for i in range(0, len(file_content), max_chunk_size)]
    collected_responses = []

    for chunk in chunks:
        response = send_request(chunk, question, url, headers)
        if response:
            collected_responses.append(response)

    combined_response = " ".join(collected_responses)

    if len(combined_response) > max_chunk_size:
        return process_large_content(combined_response, question)  # Recursive call

    return combined_response  # Return final summarized response

#using the model with its API
def send_request(file_content: str, question: str, url: str, headers: dict) -> str:
    """Helper function to send API request."""
    messages = f'''Here is a document containing relevant text.
        Use the extracted text to answer my question.
        <Document>
        {file_content}
        </Document>

        **Answer my question:**
        <Question>{question}</Question>'''

    data = {"inputs": messages, "parameters": {"max_new_tokens": 500}}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return str(response.json()[0].get('generated_text') ) 
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")