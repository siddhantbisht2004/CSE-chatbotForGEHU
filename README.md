# CSE Department Chatbot 🤖

An intelligent document-based chatbot designed to help students retrieve academic information from CSE department documents using Natural Language Processing (NLP) and Semantic Search.

## 🌐 Live Demo

**Try the chatbot here:** [https://siddhantbisht2004-cse-chatbot-for-gehu.hf.space/](YOUR_LIVE_LINK)

## ✨ Features

- Upload PDF, DOCX, and TXT documents.
- Automatic text extraction and chunking.
- Semantic search using sentence embeddings.
- Similarity matching using cosine similarity.
- Retrieves the top 3 relevant document passages.
- JSON-based knowledge-base storage.
- REST APIs built with FastAPI.

## 🛠️ Tech Stack

- **Language:** Python
- **Backend:** FastAPI
- **NLP:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Similarity Search:** Scikit-learn, Cosine Similarity
- **Libraries:** NumPy, PyPDF2, python-docx
- **Storage:** JSON

## ⚙️ How It Works

1. User uploads a PDF, DOCX, or TXT document.
2. The backend extracts text and divides it into smaller chunks.
3. Sentence Transformers converts chunks into numerical embeddings.
4. Embeddings and document information are stored in a JSON knowledge base.
5. User submits a question.
6. The question is converted into an embedding and compared with stored embeddings using cosine similarity.
7. The system retrieves and returns the most relevant passages.

## 🏗️ Architecture

- **main.py:** Handles FastAPI application and API endpoints.
- **chatbot_core.py:** Handles document processing, embedding generation, and semantic search.
- **Knowledge Base:** Stores document chunks, metadata, and embeddings in JSON format.

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/upload` | Upload documents |
| POST | `/query` | Ask questions |
| GET | `/status` | Check chatbot status |
| GET | `/health` | Health check |
| DELETE | `/knowledge-base` | Clear knowledge base |
| POST | `/reload-knowledge-base` | Reload stored data |

Interactive API documentation is available at `/docs`.

## 🚀 Future Improvements

- LLM integration for natural-language answers.
- Conversational frontend interface.
- User authentication and authorization.
- Vector database integration.
- Source citations for retrieved information.

---

**Developed as an academic project at Graphic Era Hill University, Dehradun Campus.**
