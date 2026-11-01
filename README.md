---
title: CSE Chatbot For GEHU
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# CSE Chatbot

A document-based CSE department question-answering chatbot built with FastAPI and sentence-transformers.

## Run locally

```bash
python -m pip install -r requirements.txt
python main.py
```

Open [http://localhost:8000](http://localhost:8000) for the chat interface or [http://localhost:8000/docs](http://localhost:8000/docs) for the API.

The embedding model downloads the first time the app starts. You can upload PDF, DOCX, and TXT documents from the API; uploads and generated knowledge-base changes are stored on the server's local filesystem.

## Deploy on Hugging Face Spaces

Create a **Docker** Space, then push this repository to it. The included `Dockerfile` serves the app on port `7860`. The first startup downloads the `all-MiniLM-L6-v2` embedding model and may take a few minutes. Free Spaces can pause after inactivity, and files uploaded at runtime may not persist across restarts.

The interactive chat UI is served at `/`, with API documentation at `/docs`.
