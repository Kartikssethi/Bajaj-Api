import os
import tempfile
import requests
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from unstructured.partition.pdf import partition_pdf
import google.generativeai as genai

# Fix OpenMP issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()

# === Gemini API Key ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "your_actual_key_here"
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env or manually")
genai.configure(api_key=GEMINI_API_KEY)

# === App and FAISS Setup ===
app = FastAPI()
EMBEDDING_DIM = 768
FAISS_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)
FAISS_ID_TO_TEXT = {}

# === Kartik Directory Setup ===
KARTIK_DIR = os.path.join(os.path.dirname(__file__), '..', 'kartik')
KARTIK_DIR = os.path.abspath(KARTIK_DIR)
os.makedirs(KARTIK_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(KARTIK_DIR, "faiss_index.index")
FAISS_TEXT_MAP_PATH = os.path.join(KARTIK_DIR, "faiss_id_to_text.pkl")

# === Save/Load Utility Functions ===
def save_faiss_data():
    faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
    with open(FAISS_TEXT_MAP_PATH, "wb") as f:
        pickle.dump(FAISS_ID_TO_TEXT, f)

def load_faiss_data():
    global FAISS_INDEX, FAISS_ID_TO_TEXT
    if os.path.exists(FAISS_INDEX_PATH):
        FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
    else:
        FAISS_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)

    if os.path.exists(FAISS_TEXT_MAP_PATH):
        with open(FAISS_TEXT_MAP_PATH, "rb") as f:
            FAISS_ID_TO_TEXT = pickle.load(f)
    else:
        FAISS_ID_TO_TEXT = {}

# Load on app startup
load_faiss_data()

# === Embedding ===
def get_embedding(text):
    response = genai.embed_content(model="models/embedding-001", content=text)
    return response.get("embedding", [])

# === Upload ===
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    file_path = f"./tmp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            chunking_strategy="by_title",
            infer_table_strategy=True,
            max_characters=1000,
            new_after_n_chars=1200,
            combine_text_under_n_chars=300,
        )
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            embedding = get_embedding(text)
            if not embedding:
                continue
            embedding_np = np.array(embedding, dtype='float32').reshape(1, -1)
            idx = FAISS_INDEX.ntotal
            FAISS_INDEX.add(embedding_np)
            FAISS_ID_TO_TEXT[idx] = text
        save_faiss_data()
        return {"message": "File processed and embeddings stored."}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# === Ask ===
@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")

    q_emb = get_embedding(question)
    if not q_emb:
        raise HTTPException(status_code=500, detail="Failed to get embedding for question")

    q_emb = np.array(q_emb, dtype='float32').reshape(1, -1)
    if FAISS_INDEX.ntotal == 0:
        raise HTTPException(status_code=400, detail="No data in FAISS index")

    D, I = FAISS_INDEX.search(q_emb, k=1)
    idx = int(I[0][0])
    context = FAISS_ID_TO_TEXT.get(idx, '')

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
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    response = model.generate_content(prompt)
    return {"answer": response.text.strip()}

# === HackRx Run ===
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

        elements = partition_pdf(
            filename=tmp_file_path,
            strategy="hi_res",
            chunking_strategy="by_title",
            infer_table_strategy=True,
            max_characters=1000,
            new_after_n_chars=1200,
            combine_text_under_n_chars=300,
        )
        global FAISS_INDEX, FAISS_ID_TO_TEXT
        FAISS_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)
        FAISS_ID_TO_TEXT = {}
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            embedding = get_embedding(text)
            if not embedding:
                continue
            embedding_np = np.array(embedding, dtype='float32').reshape(1, -1)
            idx = FAISS_INDEX.ntotal
            FAISS_INDEX.add(embedding_np)
            FAISS_ID_TO_TEXT[idx] = text
        save_faiss_data()
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    answers = []
    for question in questions:
        q_emb = get_embedding(question)
        if not q_emb or FAISS_INDEX.ntotal == 0:
            answers.append("")
            continue
        q_emb = np.array(q_emb, dtype='float32').reshape(1, -1)
        D, I = FAISS_INDEX.search(q_emb, k=min(3, FAISS_INDEX.ntotal))
        context_chunks = [FAISS_ID_TO_TEXT.get(int(idx), '') for idx in I[0] if int(idx) in FAISS_ID_TO_TEXT]
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
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content(prompt)
        answers.append(response.text.strip())

    return {"answers": answers}
