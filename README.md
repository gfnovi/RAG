# RAG
Rag (Retreival Augmented Generation) Python solution with llama3, LangChain, Ollama and ChromaDB in a Flask API based solution


## Overview
This application is built using Flask and integrates LangChain and other AI tools to provide two main functionalities:

1. Querying general-purpose AI (via Ollama's "llama3.2" model).
2. Uploading, storing, and querying PDF documents using vector search and embeddings.

The app exposes several endpoints to perform these actions, allowing users to interact with AI models and retrieve context-specific answers from uploaded PDF documents.

## Requirements

### Python Packages
- `Flask`
- `langchain_community`
- `langchain`
- `PDFPlumber`
- `Chroma`

### Installation
```bash
pip install flask langchain_community langchain pdfplumber chromadb
```

### Files and Folders
- **`db/`**: Directory to persist Chroma vector store data.
- **`pdf/`**: Directory to store uploaded PDFs.

## Endpoints

### 1. **Query General AI**
#### Endpoint:
`POST /askAi`
#### Description:
Sends a query to the Ollama AI model and retrieves a response.
#### Request:
```json
{
    "query": "Your question here"
}
```
#### Response:
```json
{
    "answer": "The AI's response"
}
```

---

### 2. **Query PDF Documents**
#### Endpoint:
`POST /askPdf`
#### Description:
Queries the uploaded PDFs for answers based on the provided context using a vector store and AI model.
#### Request:
```json
{
    "query": "Your question related to the PDFs"
}
```
#### Response:
```json
{
    "answer": "The AI's response",
    "sources": [
        {
            "source": "File name and metadata",
            "page_content": "Relevant content from the PDF"
        }
    ]
}
```

---

### 3. **Upload a PDF**
#### Endpoint:
`POST /addPdf`
#### Description:
Uploads and processes a PDF document. Splits the document into chunks, creates embeddings, and stores them in the Chroma vector store.
#### Request:
- Multipart form-data with the key `file` containing the PDF file.
#### Response:
```json
{
    "status": "Successfully Uploaded",
    "filename": "Uploaded file name",
    "doc_len": 10,
    "chunks": 25
}
```

---

### 4. **Test API Availability**
#### Endpoint:
`GET /ask`
#### Description:
Returns a simple static response to confirm the API is running.
#### Response:
```
asd
```

## Key Components

### 1. **AI Model (Ollama)**
- The `Ollama` model, configured with "llama3.2:latest," processes general queries and PDF-based questions.

### 2. **Vector Store (Chroma)**
- Chroma is used to store and retrieve PDF content. Embeddings are generated using `FastEmbedEmbeddings`, and documents are split into chunks for efficient processing using `RecursiveCharacterTextSplitter`.

### 3. **PDF Processing**
- The `PDFPlumberLoader` extracts text from PDFs, and the extracted content is stored in chunks for embedding and retrieval.

### 4. **Prompt Template**
- A custom prompt template is defined for retrieving context-aware answers from the model.

## Running the Application
Run the app in debug mode on port 5000:
```bash
python app.py
```

The application will be accessible at `http://127.0.0.1:5000`.

## Folder Structure
```plaintext
.
├── app.py               # Main application file
├── db/                  # Persistent Chroma vector store
├── pdf/                 # Uploaded PDF documents
└── README.md            # Documentation
```

## Notes
- Ensure the required folders (`db/` and `pdf/`) exist before running the app.
- PDFs uploaded are stored in the `pdf/` folder and processed into the vector store.

## Future Improvements
- Add authentication and rate-limiting to endpoints.
- Enhance error handling and logging.
- Support additional document formats (e.g., Word, plain text).
- Improve scalability for larger document databases.

