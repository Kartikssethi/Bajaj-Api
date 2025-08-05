# server.py

import io, asyncio
from typing import List
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import aiohttp
from PyPDF2 import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# === Config ===
genai.configure(api_key="AIzaSyAtu0Gxg4_yQlp0hHl6JTCoDqQioKNW0RI")
GEMINI_MODEL = "gemini-2.5-flash"           # GA thinking-enabled model :contentReference[oaicite:6]{index=6}
# or use "gemini-2.5-flash-lite" for speed/cost efficiency :contentReference[oaicite:7]{index=7}
THINK_BUDGET = 512  # optional thinking control; lower means faster

EMBED_MODEL = SentenceTransformer("intfloat/e5-small-v2")

app = FastAPI()

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

async def download_pdf(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        resp = await session.get(url)
        if resp.status != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch document")
        return await resp.read()

def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text: str, size: int = 500) -> List[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]

def embed_chunks(chunks: List[str]) -> np.ndarray:
    return EMBED_MODEL.encode(chunks, normalize_embeddings=True)

def semantic_search(question: str, chunk_embeds: np.ndarray, chunks: List[str], top_k=3) -> List[str]:
    q_vec = EMBED_MODEL.encode([question], normalize_embeddings=True)[0]
    index = faiss.IndexFlatIP(q_vec.shape[0])
    index.add(np.array(chunk_embeds).astype("float32"))
    _, I = index.search(np.array([q_vec], dtype="float32"), top_k)
    return [chunks[i] for i in I[0]]

async def call_gemini(prompt: str) -> str:
    resp = genai.Client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        thinking_config={"max_tokens_for_thinking": THINK_BUDGET}
    )
    return resp.text

@app.post("/hackrx/run", response_model=QAResponse)
async def handle(payload: QARequest, authorization: str = Header(None)):
    # Optional: validate authorization header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid auth token")

    pdf_bytes = await download_pdf(payload.documents)
    text = extract_text(pdf_bytes)
    chunks = chunk_text(text)
    embeds = embed_chunks(chunks)

    prompts = []
    for q in payload.questions:
        context = semantic_search(q, embeds, chunks)
        prompt = "Context:\n" + "\n\n".join(context) + "\n\nQuestion: " + q
        prompts.append(prompt)

    answers = []
    # batch in groups to limit LLM calls and avoid rate limiting
    for i in range(0, len(prompts), 5):
        batch = prompts[i:i+5]
        results = await asyncio.gather(*(asyncio.to_thread(call_gemini, p) for p in batch))
        answers.extend(results)

    return QAResponse(answers=answers)