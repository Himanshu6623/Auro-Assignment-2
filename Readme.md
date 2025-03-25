# FastAPI Embedding Storage and Retrieval System  

This project is a FastAPI-based application that processes text files and stores their embeddings in a PostgreSQL database. It also provides an API to retrieve relevant text based on user queries using a Hugging Face model.  

## Features  
- Upload `.txt` files and store embeddings in a PostgreSQL database.  
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`.  
- Store a combined embedding of all uploaded files for global retrieval.  
- Retrieve relevant text using Hugging Face’s `gemma-3-27b-it` model.  
- Support for querying individual files or the combined database.  

## Installation  

### Prerequisites  
- Python 3.8+  
- PostgreSQL installed and running  

### Clone the Repository  
```sh  
git clone https://github.com/your-repo.git  
cd your-repo  
```  

### Install Dependencies  
```sh  
pip install -r requirements.txt  
```  

### Set Up PostgreSQL  
Update `DATABASE_URL` in the script:  
```python  
DATABASE_URL = "postgresql+asyncpg://<username>:<password>@localhost:5432/<database_name>"  
```  

Create the database and initialize tables:  
```sh  
python -c "import asyncio; from your_script import init_db; asyncio.run(init_db())"  
```  

## Running the API  
```sh  
uvicorn main:app --host 0.0.0.0 --port 8000 --reload  
```  

## API Endpoints  

### Upload File and Generate Embedding  
**Endpoint:** `POST /uploadfile/`  
**Description:** Uploads a `.txt` file, generates its embedding, and stores it in the database.  
**Request:**  
```json  
{ "file": "example.txt" }  
```  
**Response:**  
```json  
{ "message": "File 'example.txt' processed and stored successfully" }  
```  

### Upload Text and Generate Embedding  
**Endpoint:** `POST /uploadText/`  
**Description:** Directly uploads text content, generates its embedding, and stores it.  
**Request:**  
```json  
{ "value": "Sample text content." }  
```  
**Response:**  
```json  
{ "message": "Text processed and stored successfully", "filename": "text_<hash>.txt" }  
```  

### Retrieve Relevant Text  
**Endpoint:** `GET /retrieve/`  
**Description:** Retrieves the most relevant text from a stored file based on a query.  
**Request:**  
```json  
{ "filename": "example.txt", "question": "What is this document about?" }  
```  
**Response:**  
```json  
{ "relevant_text": "Extracted relevant content..." }  
```  

## Database Schema  

### Table: `embeddings`  
| Column Name | Type | Description |  
|-------------|--------|-------------|  
| filename | String | Primary Key (unique file name) |  
| embedding | LargeBinary | Serialized embedding vector |  
| content | TEXT | Original document text |  

## Logging  
The app logs important events using Python’s logging module. Logs include:  
- Server startup and shutdown  
- File processing status  
- Query execution details  

## Error Handling  
- Invalid file formats return a `400 Bad Request`.  
- Empty files or text return a `400 Bad Request`.  
- Internal processing errors return a `500 Internal Server Error`.  

## License  
MIT License  
