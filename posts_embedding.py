import os
import json
import tiktoken
import requests
from dotenv import load_dotenv

# === Config ===
AI_PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
load_dotenv()
PROXY_TOKEN = os.getenv("PROXY_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {PROXY_TOKEN}",
    "Content-Type": "application/json"
}

DISCOURSE_FILE = "discourse_posts.json"
EMBEDDINGS_FILE = "discourse_posts_embeddings.json"
MODEL_NAME = "text-embedding-3-small"

tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 10000

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
    with open(DISCOURSE_FILE, "r", encoding="utf-8") as f:
        posts = json.load(f)

    embeddings = []

    for post in posts:
        content = post.get("content", "")
        if not content.strip():
            continue

        chunks = chunk_text(content)
        print(f"Processing post ID {post.get('post_id')} with {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            embeddings.append({
                "post_id": post.get("id"),
                "chunk_index": i,
                "text": chunk,
                "embedding": emb,
                "url": post.get("url")
            })

    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f_out:
        json.dump(embeddings, f_out)

    print(f"✅ Saved {len(embeddings)} chunk embeddings to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    main()
