import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

def ask_ollama(question, context=""):
    prompt = f"""You are an AI data scientist.
Answer the user's question using the supplied dataset/model context.
Do not invent metrics or facts. If information is missing, say so.

CONTEXT:
{context}

QUESTION:
{question}
"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as exc:
        return f"Local Ollama is unavailable. Start Ollama and run '{OLLAMA_MODEL}'. Details: {exc}"
