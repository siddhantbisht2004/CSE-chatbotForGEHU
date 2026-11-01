"""
FastAPI server for CSE Document Question Answer Chatbot
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import numpy as np
from typing import List, Dict, Optional
import logging
import tempfile
import shutil

from chatbot_core import DocumentProcessor, CSEChatbot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import os
import json
import numpy as np
from typing import List, Dict, Optional
import logging
import tempfile
import shutil

from chatbot_core import DocumentProcessor, CSEChatbot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot: Optional[CSEChatbot] = None
KNOWLEDGE_BASE_PATH = "knowledge_base.json"
UPLOAD_DIR = "./uploaded_documents"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global chatbot
    logger.info("Initializing chatbot...")
    chatbot = CSEChatbot()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        logger.info("Loading existing knowledge base...")
        chatbot.load_knowledge_base(KNOWLEDGE_BASE_PATH)
        logger.info(f"Loaded {len(chatbot.documents)} document chunks from knowledge base")
    else:
        logger.info("No existing knowledge base found. Please upload documents.")
    yield
    logger.info("Shutting down chatbot...")


# Initialize FastAPI app with modern lifespan
app = FastAPI(
    title="CSE Chatbot API",
    description="Document-based Question Answering System",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    query: str
    response: str
    relevant_documents: List[Dict]
    processing_time: float


class StatusResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CSE Department AI Assistant</title>
  <style>
    :root {
      --primary: #4f46e5;
      --primary-hover: #4338ca;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      justify-content: center;
      height: 100vh;
      padding: 16px;
    }
    .app-container {
      width: 100%;
      max-width: 900px;
      background: var(--card-bg);
      border-radius: 16px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    header {
      background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
      color: white;
      padding: 18px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .header-title h1 { font-size: 1.25rem; font-weight: 700; }
    .header-title p { font-size: 0.85rem; opacity: 0.9; margin-top: 2px; }
    .header-links a {
      background: rgba(255, 255, 255, 0.15);
      color: white;
      text-decoration: none;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 600;
      transition: background 0.2s;
    }
    .header-links a:hover { background: rgba(255, 255, 255, 0.3); }
    .status-bar {
      background: #eff6ff;
      border-bottom: 1px solid #dbeafe;
      padding: 8px 24px;
      font-size: 0.82rem;
      color: #1e40af;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 600;
    }
    .dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; }
    .chat-box {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .message {
      max-width: 82%;
      padding: 14px 18px;
      border-radius: 14px;
      font-size: 0.92rem;
      line-height: 1.5;
      animation: fadeIn 0.2s ease-in-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .message.bot {
      align-self: flex-start;
      background: #f1f5f9;
      color: var(--text);
      border-bottom-left-radius: 4px;
    }
    .message.user {
      align-self: flex-end;
      background: var(--primary);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .meta-box {
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px dashed #cbd5e1;
      font-size: 0.78rem;
      color: var(--text-muted);
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .meta-chip {
      background: #ffffff;
      padding: 2px 8px;
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .suggestions {
      padding: 8px 24px 12px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .suggestion-btn {
      background: #f8fafc;
      border: 1px solid var(--border);
      color: #334155;
      padding: 6px 12px;
      border-radius: 16px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .suggestion-btn:hover {
      background: #eef2ff;
      border-color: #c7d2fe;
      color: var(--primary);
    }
    .input-container {
      padding: 16px 24px;
      border-top: 1px solid var(--border);
      display: flex;
      gap: 10px;
      background: #ffffff;
    }
    .input-container input {
      flex: 1;
      padding: 12px 16px;
      border: 1px solid var(--border);
      border-radius: 10px;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }
    .input-container input:focus { border-color: var(--primary); }
    .input-container button {
      background: var(--primary);
      color: white;
      border: none;
      padding: 12px 22px;
      border-radius: 10px;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    .input-container button:hover { background: var(--primary-hover); }
    pre { white-space: pre-wrap; word-break: break-word; font-family: inherit; }
  </style>
</head>
<body>
  <div class="app-container">
    <header>
      <div class="header-title">
        <h1>CSE Department AI Chatbot</h1>
        <p>Document-based Semantic Retrieval System</p>
      </div>
      <div class="header-links">
        <a href="/docs" target="_blank">Swagger API Docs ↗</a>
      </div>
    </header>

    <div class="status-bar">
      <div class="status-badge">
        <span class="dot"></span>
        <span id="status-text">Knowledge base active (7 chunks loaded)</span>
      </div>
      <span style="opacity: 0.75;">Embedding Model: all-MiniLM-L6-v2</span>
    </div>

    <div class="chat-box" id="chat-box">
      <div class="message bot">
        <strong>Assistant:</strong><br>
        Hello! I am your CSE Department Assistant. You can ask me anything about department courses, prerequisites, faculty profiles, and research areas!
      </div>
    </div>

    <div class="suggestions">
      <button class="suggestion-btn" onclick="sendSuggestion('Tell me about CS101')">📘 Tell me about CS101</button>
      <button class="suggestion-btn" onclick="sendSuggestion('Who is Dr. Sarah Johnson?')">👩‍🏫 Who is Dr. Sarah Johnson?</button>
      <button class="suggestion-btn" onclick="sendSuggestion('What are the prerequisites for Data Structures?')">🔍 Prerequisites for Data Structures</button>
      <button class="suggestion-btn" onclick="sendSuggestion('What research areas are available?')">🔬 Research areas</button>
    </div>

    <div class="input-container">
      <input type="text" id="user-input" placeholder="Ask a question about the CSE department..." onkeydown="handleKey(event)" />
      <button onclick="sendMessage()">Send</button>
    </div>
  </div>

  <script>
    async function loadStatus() {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('status-text').innerText = `Knowledge base: ${data.documents_loaded} document chunks ready`;
      } catch (e) {
        console.error(e);
      }
    }
    loadStatus();

    function handleKey(e) {
      if (e.key === 'Enter') sendMessage();
    }

    function sendSuggestion(text) {
      document.getElementById('user-input').value = text;
      sendMessage();
    }

    async function sendMessage() {
      const input = document.getElementById('user-input');
      const query = input.value.trim();
      if (!query) return;

      const chatBox = document.getElementById('chat-box');

      // Add user message
      const userDiv = document.createElement('div');
      userDiv.className = 'message user';
      userDiv.innerText = query;
      chatBox.appendChild(userDiv);

      input.value = '';
      chatBox.scrollTop = chatBox.scrollHeight;

      // Add loading indicator
      const botDiv = document.createElement('div');
      botDiv.className = 'message bot';
      botDiv.innerHTML = '<em>Searching knowledge base...</em>';
      chatBox.appendChild(botDiv);
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const response = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query, top_k: 3 })
        });
        const data = await response.json();

        let metaHtml = '';
        if (data.relevant_documents && data.relevant_documents.length > 0) {
          const topDoc = data.relevant_documents[0];
          const score = (topDoc.similarity * 100).toFixed(1);
          const source = topDoc.metadata ? topDoc.metadata.source.split(/[\\\\/]/).pop() : 'document';
          metaHtml = `
            <div class="meta-box">
              <span class="meta-chip">⚡ Score: ${score}% match</span>
              <span class="meta-chip">📄 Source: ${source}</span>
              <span class="meta-chip">⏱️ ${data.processing_time.toFixed(3)}s</span>
            </div>
          `;
        }

        botDiv.innerHTML = `<strong>Assistant:</strong><br><pre>${data.response}</pre>${metaHtml}`;
      } catch (err) {
        botDiv.innerHTML = `<strong>Error:</strong> Could not retrieve response from server.`;
      }
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
async def root():
    """Interactive Web Chatbot UI"""
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api", tags=["Root"])
async def api_info():
    """API overview endpoints"""
    return {
        "message": "CSE Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "web_ui": "/",
            "docs": "/docs",
            "status": "/status",
            "query": "/query",
            "upload": "/upload",
            "health": "/health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot is not None
    }


@app.get("/status", tags=["Status"], response_model=StatusResponse)
async def status():
    """Get chatbot status"""
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    documents_count = len(chatbot.documents) if chatbot.documents else 0
    
    return StatusResponse(
        status="ready" if documents_count > 0 else "no_documents",
        message=f"Knowledge base contains {documents_count} document chunks" if documents_count > 0 
                else "No documents loaded. Please upload documents first.",
        documents_loaded=documents_count
    )


@app.post("/upload", tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Upload documents (PDF, DOCX, TXT) and process them
    """
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create temporary directory for uploads
    temp_dir = tempfile.mkdtemp()
    uploaded_files = []
    
    try:
        # Save uploaded files
        for file in files:
            if file.filename:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                uploaded_files.append({
                    "filename": file.filename,
                    "size": len(content)
                })
        
        # Process documents in background
        background_tasks.add_task(
            process_and_save_documents,
            temp_dir,
            uploaded_files
        )
        
        return {
            "status": "processing",
            "message": f"Received {len(uploaded_files)} file(s). Processing in background...",
            "files": uploaded_files
        }
    
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


def process_and_save_documents(directory: str, uploaded_files: List[Dict]):
    """
    Process uploaded documents and save knowledge base
    This runs in the background
    """
    try:
        logger.info(f"Processing {len(uploaded_files)} documents...")
        
        # Process documents
        docs = DocumentProcessor.process_directory(directory)
        chatbot.documents.extend(docs)
        
        # Generate embeddings
        if chatbot.documents:
            texts = [doc["content"] for doc in chatbot.documents]
            chatbot.embeddings = chatbot.encoder.encode(texts, show_progress_bar=False)
        
        # Save knowledge base
        chatbot.save_knowledge_base(KNOWLEDGE_BASE_PATH)
        
        logger.info(f"Successfully processed and saved {len(docs)} document chunks")
    
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if os.path.exists(directory):
            shutil.rmtree(directory)


@app.post("/query", tags=["Query"], response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Submit a query to the chatbot
    """
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    if len(chatbot.documents) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first."
        )
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = chatbot.get_response(request.query, top_k=request.top_k)
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/knowledge-base", tags=["Documents"])
async def delete_knowledge_base():
    """
    Clear the knowledge base
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        chatbot.documents = []
        chatbot.embeddings = None
        
        if os.path.exists(KNOWLEDGE_BASE_PATH):
            os.remove(KNOWLEDGE_BASE_PATH)
        
        return {
            "status": "success",
            "message": "Knowledge base cleared"
        }
    
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-knowledge-base", tags=["Documents"])
async def reload_knowledge_base():
    """
    Reload knowledge base from file
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        raise HTTPException(status_code=400, detail="Knowledge base file not found")
    
    try:
        chatbot.load_knowledge_base(KNOWLEDGE_BASE_PATH)
        return {
            "status": "success",
            "message": f"Knowledge base reloaded with {len(chatbot.documents)} documents"
        }
    
    except Exception as e:
        logger.error(f"Error reloading knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
