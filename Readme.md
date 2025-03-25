FastAPI Embedding Storage and Retrieval

This FastAPI-based project enables users to upload text files or raw text, generate embeddings using the sentence-transformers/all-MiniLM-L6-v2 model, store them in a PostgreSQL database, and retrieve relevant text based on queries. Additionally, it utilizes Hugging Face's google/gemma-3-27b-it API for processing large text inputs and answering questions.

Features

Upload text files (.txt) and generate embeddings

Store embeddings in PostgreSQL with asyncpg

Retrieve stored content using filename or aggregated embedding

Process large text documents efficiently

Use Hugging Face API for answering questions

Installation

Prerequisites

Python 3.8+

PostgreSQL database

Hugging Face API key

Setup

Clone the repository:

Create and activate a virtual environment:

Install dependencies:

Configuration

Environment Variables

Set up the following variables in your .env file or directly in the script:

Running the Application

Start FastAPI Server

API Endpoints

1. Upload a Text File

Endpoint: POST /uploadfile/

Description: Uploads a .txt file, generates embeddings, and stores it.

Request:

Response:

2. Upload Raw Text

Endpoint: POST /uploadText/

Description: Accepts raw text, generates embeddings, and stores them.

Request:

Response:

3. Retrieve Relevant Text

Endpoint: GET /retrieve/

Description: Retrieves relevant text from a stored file or the combined dataset.

Request:

Response:

Database Schema

The project uses SQLAlchemy ORM with the following schema:

EmbeddingModel:

filename (String): Primary key

embedding (LargeBinary): Serialized embedding vector

content (TEXT): Stored text content

Processing Large Text

For large documents exceeding 5000 characters, the text is split into chunks, processed separately, and merged recursively.

API Integration

This project integrates with Hugging Face's google/gemma-3-27b-it API to generate responses based on stored text.

API Request: Sends a formatted prompt with stored content and the user’s question.

API Response: Extracts and returns the relevant answer.

Logging & Debugging

Logging is configured using Python’s logging module. Errors are logged, and meaningful messages are displayed for debugging.

Contributing

Fork the repository

Create a new branch

Make your changes and submit a pull request
