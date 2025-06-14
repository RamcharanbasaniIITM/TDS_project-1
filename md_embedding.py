import os
import json
import glob
import tiktoken  # for token counting
import requests
from dotenv import load_dotenv
# === Config ===
AI_PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"  # Correct proxy endpoint
load_dotenv()
PROXY_TOKEN = os.getenv("PROXY_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {PROXY_TOKEN}",
    "Content-Type": "application/json"
}

MARKDOWN_DIR = "markdown_files"
EMBEDDINGS_FILE = "embeddings.json"
MODEL_NAME = "text-embedding-3-small"  # Supported by AI Proxy

# Load tokenizer to count tokens (OpenAI-compatible)
tokenizer = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 1000

def chunk_text(text, max_tokens=MAX_TOKENS):
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

    for word in words:
        tokens = tokenizer.encode(word)
        if current_len + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = len(tokens)
        else:
            current_chunk.append(word)
            current_len += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_embedding(text):
    payload = {
        "model": MODEL_NAME,
        "input": text
    }
    response = requests.post(AI_PROXY_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]

def main():
    embeddings = []
    files = glob.glob(os.path.join(MARKDOWN_DIR, "*.md"))
    print(f"Found {len(files)} markdown files")

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_text(content)
        print(f"Processing {os.path.basename(filepath)} with {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            embeddings.append({
                "filename": os.path.basename(filepath),
                "chunk_index": i,
                "text": chunk,
                "embedding": emb
            })

    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print(f"✅ Saved {len(embeddings)} chunk embeddings to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    main()
