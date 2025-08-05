from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import faiss
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import requests
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from google import genai

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# FastAPI app
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=5)

# Request and Response Models
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# Prompt builder with k-shot examples
def build_batch_prompt(context, questions: List[str]) -> str:
    question_lines = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return f"""
You are an LLM-powered assistant for HackRx 6.0, built by Bajaj Finserv Health. Answer user questions strictly based on the given CONTEXT.

✅ RULES:
1. Use **only** the information from CONTEXT.
2. Do **not** guess or hallucinate.
3. Keep each answer informative yet concise (max 3 sentences).
4. Respond with only direct answers in numbered list.
5. If unsure, say: "Not enough info from document."

---
📘 CONTEXT:
{context}

---
❓ QUESTIONS:
{question_lines}

---
📝 ANSWERS:
"""

# Async Gemini call
def generate_answer(prompt):
    try:
        result =   client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return result.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Main route
@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_hackrx(request: RunRequest, authorization: Optional[str] = Header(None)):
    if authorization != "Bearer d2616659a7fd02e3f14cae15e7660eac6d24581d4e9f5dcecd8ac5cf55db138a":
        raise HTTPException(status_code=401, detail="Unauthorized")

    response = requests.get(request.documents)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download document")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path, mode="page")
    documents = loader.load()
    for idx, doc in enumerate(documents):
        doc.metadata["source_page"] = idx + 1

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    documents = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(documents, embeddings)


    all_contexts = []
    for question in request.questions:
        docs = vectorstore.similarity_search(question, k=2)
        chunk = "\n\n".join([
            f"(From Page {doc.metadata.get('source_page', 'N/A')})\n{doc.page_content}"
            for doc in docs
        ])
        all_contexts.append(chunk)

    combined_context = "\n\n".join(all_contexts)
    prompt = build_batch_prompt(combined_context, request.questions)
    print("\n📝 Batched Prompt:\n", prompt)

    final_answer = generate_answer(prompt)

    split_answers = [line.split(". ", 1)[-1].strip() for line in final_answer.strip().split("\n") if line.strip()]

    return RunResponse(answers=split_answers)

