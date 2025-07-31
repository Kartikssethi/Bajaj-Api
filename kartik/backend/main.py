import os
import tempfile
import requests
import pickle
import numpy as np
import faiss
import redis
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
import google.generativeai as genai

# Fix OpenMP issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

# Load Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")
genai.configure(api_key=GEMINI_API_KEY)

# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# FAISS setup
EMBEDDING_DIM = 768
KARTIK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kartik'))
os.makedirs(KARTIK_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(KARTIK_DIR, "faiss_index.index")
MAPPING_PATH = os.path.join(KARTIK_DIR, "faiss_id_to_text.pkl")

if os.path.exists(FAISS_INDEX_PATH):
    FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
else:
    FAISS_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)

if os.path.exists(MAPPING_PATH):
    with open(MAPPING_PATH, "rb") as f:
        FAISS_ID_TO_TEXT = pickle.load(f)
else:
    FAISS_ID_TO_TEXT = {}

# FastAPI app
app = FastAPI()

# Embedding helper
def get_embedding(text):
    cache_key = f"embedding:{hash(text)}"
    cached = redis_client.get(cache_key)
    if cached:
        return np.frombuffer(cached, dtype='float32')
    response = genai.embed_content(model="text-embedding-004", content=text)
    embedding = np.array(response.get("embedding", []), dtype='float32')
    redis_client.set(cache_key, embedding.tobytes())
    return embedding

# Extract 10-page chunks
def extract_chunks_by_10_pages(file_path):
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    chunks = []
    for i in range(0, total_pages, 5):
        chunk = ""
        for j in range(i, min(i + 5, total_pages)):
            page_text = reader.pages[j].extract_text()
            if page_text:
                chunk += page_text
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

# Save FAISS + mapping
def persist_faiss_data():
    faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(FAISS_ID_TO_TEXT, f)

# Upload PDF
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    file_path = f"./tmp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    try:
        chunks = extract_chunks_by_10_pages(file_path)
        for text in chunks:
            embedding = get_embedding(text).reshape(1, -1)
            idx = FAISS_INDEX.ntotal
            FAISS_INDEX.add(embedding)
            FAISS_ID_TO_TEXT[idx] = text
        persist_faiss_data()
        return {"message": "File processed and embeddings stored."}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# Ask question
@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    if FAISS_INDEX.ntotal == 0:
        raise HTTPException(status_code=400, detail="FAISS index is empty")
    q_emb = get_embedding(question).reshape(1, -1)
    D, I = FAISS_INDEX.search(q_emb, k=min(3, FAISS_INDEX.ntotal))
    context_chunks = [FAISS_ID_TO_TEXT.get(int(i), "") for i in I[0]]
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an expert retrieval assistant for insurance policy documents. Your task is to read through provided policy excerpts and answer user questions based strictly on the content.

Guidelines:
- Only use the provided context for your answer.
- Be specific, concise, and explain any relevant conditions or limitations.
- If the information is not present in the context, clearly say so.
- Include a short explanation if needed.
- Output should be in the following JSON format:
{{
  "answer": "...",
  "source_clause": "...",
  "reasoning": "..."
}}

Context:
{context}

Question:
{question}
"""
    model = genai.GenerativeModel(model_name="gemini-2.5-pro")
    response = model.generate_content(prompt)
    return {"answer": response.text.strip()}

# HackRx run endpoint
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
        chunks = extract_chunks_by_10_pages(tmp_file_path)
        global FAISS_INDEX, FAISS_ID_TO_TEXT
        FAISS_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)
        FAISS_ID_TO_TEXT = {}
        for text in chunks:
            embedding = get_embedding(text).reshape(1, -1)
            idx = FAISS_INDEX.ntotal
            FAISS_INDEX.add(embedding)
            FAISS_ID_TO_TEXT[idx] = text
        persist_faiss_data()
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    answers = []
    for question in questions:
        q_emb = get_embedding(question).reshape(1, -1)
        D, I = FAISS_INDEX.search(q_emb, k=min(3, FAISS_INDEX.ntotal))
        context_chunks = [FAISS_ID_TO_TEXT.get(int(i), "") for i in I[0]]
        context = "\n\n".join(context_chunks)
        prompt = f"""
You are an expert retrieval assistant for insurance policy documents. Your task is to read through provided policy excerpts and answer user questions based strictly on the content.

Guidelines:
- Only use the provided context for your answer.
- Be specific, concise, and explain any relevant conditions or limitations.
- If the information is not present in the context, clearly say so.
- Include a short explanation if needed.
- Output should be in the following JSON format:
{{
  "answer": "...",
  "source_clause": "...",
  "reasoning": "..."
}}

Context:
{context}

Question:
{question}
"""
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        response = model.generate_content(prompt)
        answers.append(response.text.strip())
    return {"answers": answers}
