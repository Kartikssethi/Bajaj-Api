import os
import tempfile
import requests
import pickle
import numpy as np
import faiss
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Load env vars
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")
genai.configure(api_key=GEMINI_API_KEY)

# Thread pool for async embeddings
executor = ThreadPoolExecutor(max_workers=5)

# FAISS
EMBEDDING_DIM = 768
KARTIK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kartik'))
os.makedirs(KARTIK_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(KARTIK_DIR, "faiss_index.index")
MAPPING_PATH = os.path.join(KARTIK_DIR, "faiss_id_to_text.pkl")

# Use cosine similarity
FAISS_INDEX = faiss.IndexFlatIP(EMBEDDING_DIM)
FAISS_ID_TO_TEXT = {}

# FastAPI
app = FastAPI()

# Normalize vectors
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# Embedding (sync)
def get_embedding(text: str) -> np.ndarray:
    response = genai.embed_content(model="text-embedding-004", content=text)
    vec = np.array(response.get("embedding", []), dtype='float32')
    return normalize(vec)

# Embedding (async)
async def get_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(executor, get_embedding, text) for text in texts]
    return await asyncio.gather(*tasks)

# Chunk text by char
def chunk_text_by_char_length(text: str, max_chars: int = 10000) -> list[str]:
    chunks = []
    while len(text) > max_chars:
        split_idx = text.rfind(".", 0, max_chars) + 1
        if split_idx == 0:
            split_idx = max_chars
        chunks.append(text[:split_idx].strip())
        text = text[split_idx:].strip()
    if text:
        chunks.append(text)
    return chunks

# Read full PDF
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

# Persist
def persist_faiss():
    faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(FAISS_ID_TO_TEXT, f)

# Upload
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    path = f"./tmp_{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        raw_text = extract_text_from_pdf(path)
        chunks = chunk_text_by_char_length(raw_text)
        embeddings = await get_embeddings_batch(chunks)
        for i, emb in enumerate(embeddings):
            emb = emb.reshape(1, -1)
            FAISS_INDEX.add(emb)
            FAISS_ID_TO_TEXT[FAISS_INDEX.ntotal - 1] = chunks[i]
        persist_faiss()
        return {"message": "Uploaded & processed"}
    finally:
        os.remove(path)

# Ask
@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    if FAISS_INDEX.ntotal == 0:
        raise HTTPException(status_code=400, detail="Empty index")
    q_emb = normalize(get_embedding(question)).reshape(1, -1)
    D, I = FAISS_INDEX.search(q_emb, k=min(3, FAISS_INDEX.ntotal))
    context_chunks = [FAISS_ID_TO_TEXT.get(int(i), "") for i in I[0]]
    context = "\n\n".join(context_chunks)

    prompt = f"""
QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. Extract the EXACT information from the document - do not say \"context doesn't contain enough information\" unless truly absent
2. Look for specific numbers, dates, percentages, conditions, and definitions
3. If you find partial information, provide what's available with specific details
4. Include specific terms, timeframes, and conditions mentioned in the document
5. For definitions, provide the complete definition as stated
6. Be comprehensive but concise - include all relevant details from the document
7. Write answers in clear, human-readable format with proper punctuation and grammar
8. Focus on factual information directly from the document

FORMAT: Respond with JSON: {{"answer": "detailed answer with specific information from document"}}

ANSWER:
Context:
{context}
"""

    model = genai.GenerativeModel(model_name="gemini-2.5-pro")
    response = model.generate_content(prompt)
    return {"answer": response.text.strip()}

# HackRx Run
@app.post("/hackrx/run")
async def hackrx_run(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    data = await request.json()
    pdf_url = data.get("documents")
    questions = data.get("questions", [])
    if not pdf_url or not questions:
        raise HTTPException(status_code=400, detail="Missing documents URL or questions")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            response = requests.get(pdf_url)
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        raw_text = extract_text_from_pdf(tmp_file_path)
        chunks = chunk_text_by_char_length(raw_text)
        embeddings = await get_embeddings_batch(chunks)

        global FAISS_INDEX, FAISS_ID_TO_TEXT
        FAISS_INDEX = faiss.IndexFlatIP(EMBEDDING_DIM)
        FAISS_ID_TO_TEXT = {}
        for i, emb in enumerate(embeddings):
            emb = emb.reshape(1, -1)
            FAISS_INDEX.add(emb)
            FAISS_ID_TO_TEXT[FAISS_INDEX.ntotal - 1] = chunks[i]
        persist_faiss()
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    answers = []
    for question in questions:
        q_emb = normalize(get_embedding(question)).reshape(1, -1)
        D, I = FAISS_INDEX.search(q_emb, k=min(3, FAISS_INDEX.ntotal))
        context_chunks = [FAISS_ID_TO_TEXT.get(int(i), "") for i in I[0]]
        context = "\n\n".join(context_chunks)

        prompt = f"""
QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. Extract the EXACT information from the document - do not say \"context doesn't contain enough information\" unless truly absent
2. Look for specific numbers, dates, percentages, conditions, and definitions
3. If you find partial information, provide what's available with specific details
4. Include specific terms, timeframes, and conditions mentioned in the document
5. For definitions, provide the complete definition as stated
6. Be comprehensive but concise - include all relevant details from the document
7. Write answers in clear, human-readable format with proper punctuation and grammar
8. Focus on factual information directly from the document

FORMAT: Respond with JSON: {{"answer": "detailed answer with specific information from document"}}

ANSWER:
Context:
{context}
"""
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        response = model.generate_content(prompt)
        answers.append(response.text.strip())

    return {"answers": answers}
