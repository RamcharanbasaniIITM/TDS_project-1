import json
import numpy as np
import requests
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from sentence_transformers import util
import os
from dotenv import load_dotenv
import re

# === Config ===
load_dotenv()
PROXY_TOKEN = os.getenv("PROXY_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {PROXY_TOKEN}",
    "Content-Type": "application/json"
}

DISCOURSE_FILE = "discourse_posts_embeddings.json"
EMBEDDINGS_FILE = "embeddings.json"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

EMBEDDING_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
CHAT_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# === Load Embeddings ===
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    content_chunks = json.load(f)

with open(DISCOURSE_FILE, "r", encoding="utf-8") as f:
    discourse_posts = json.load(f)

content_chunks = [chunk for chunk in content_chunks if chunk.get("embedding") is not None]
discourse_posts = [post for post in discourse_posts if post.get("embedding") is not None]

for chunk in content_chunks:
    chunk["embedding"] = np.array(chunk["embedding"], dtype=np.float32)

for post in discourse_posts:
    post["embedding"] = np.array(post["embedding"], dtype=np.float32)

# === FastAPI Setup ===
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: list[Link]

# Canonical link overrides
REDIRECT_OVERRIDES = {
    "https://discourse.onlinedegree.iitm.ac.in/t/ga2-deployment-tools-discussion-thread-tds-jan-2025/161120":
        "https://tds.s-anand.net/#/docker"
}

def embed_text(text: str) -> np.ndarray:
    try:
        res = requests.post(
            EMBEDDING_URL,
            headers=HEADERS,
            json={"input": text, "model": EMBEDDING_MODEL}
        )
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Embedding request failed: {e}")
    
    embedding = res.json()["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)

def get_top_k_matches(question_embedding, data, k=5):
    similarities = [
        (item, util.cos_sim(question_embedding, item["embedding"]).item())
        for item in data
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def build_context(matches):
    return "\n\n---\n\n".join(match["text"] for match, _ in matches)

def normalize_url(url):
    if not isinstance(url, str):
        return None
    match = re.match(r"(https://discourse\.onlinedegree\.iitm\.ac\.in/t/[^/]+/\d+)", url)
    if match:
        normalized = match.group(1)
        return REDIRECT_OVERRIDES.get(normalized, normalized)
    return REDIRECT_OVERRIDES.get(url, url)

def build_links(matches) -> List[Link]:
    seen = set()
    links = []

    for match, _ in matches:
        raw_url = match.get("original_url") or match.get("url")
        norm_url = normalize_url(raw_url)

        if norm_url:
            # Normalize docker doc link exactly
            if "s-anand.net" in norm_url and "docker" in norm_url:
                norm_url = "https://tds.s-anand.net/#/docker"

            if norm_url not in seen:
                links.append(Link(url=norm_url, text=match["text"][:80] + "..."))
                seen.add(norm_url)

    return links[:5]


@app.post("/api", response_model=AnswerResponse)
async def answer_question(query: Query):
    try:
        question_embedding = embed_text(query.question)

        top_content = get_top_k_matches(question_embedding, content_chunks, k=3)
        top_posts = get_top_k_matches(question_embedding, discourse_posts, k=3)
        top_all = sorted(top_content + top_posts, key=lambda x: x[1], reverse=True)[:5]

        context = build_context(top_all)

        image_data = None
        if query.image:
            if query.image.startswith("file://"):
                file_path = query.image.replace("file://", "")
                try:
                    with open(file_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                except FileNotFoundError:
                    image_data = None
            else:
                image_data = query.image

        if image_data:
            context += "\n\n---\n\nImage is attached with the question. Use it if relevant."

        payload = {
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful teaching assistant for the Tools in Data Science course at IIT Madras. "
                        "Answer student questions based strictly on the provided context. "
                        "If possible, always include a relevant link from the context (e.g., a Discourse post or course page). "
                        "Use only the URLs present in the context."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Answer the following question using the given course material and discourse posts:\n\n{context}\n\n"
                        f"Question: {query.question}"
                    )
                }
            ]
        }

        res = requests.post(CHAT_URL, headers=HEADERS, json=payload)
        res.raise_for_status()
        answer = res.json()["choices"][0]["message"]["content"].strip()

        # === Patch: Inject top source URL if missing ===
        top_links = build_links(top_all)
        if top_links:
            top_url = top_links[0].url
            if top_url not in answer:
                answer += f"\n\n[Source]({top_url})"

    except Exception as e:
        print(f"Error occurred: {e}")
        answer = "Sorry, something went wrong while processing your request."
        top_links = []

    return AnswerResponse(
        answer=answer,
        links=top_links
    )

@app.get("/")
def root():
    return {"message": "TDS Virtual TA is running"}
