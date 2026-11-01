---
title: CSE Chatbot For GEHU
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: 3.11
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

This repository includes a Gradio chat interface in `app.py` for free Hugging Face Spaces. Create a **Gradio** Space and push the project files to it. The first startup downloads the `all-MiniLM-L6-v2` embedding model and may take a few minutes. Free Spaces can pause after inactivity, and files uploaded at runtime may not persist across restarts.

The Space loads the committed `knowledge_base.json`. The original FastAPI API remains available for local use with `python main.py`; the hosted Space serves the Gradio chat interface.
