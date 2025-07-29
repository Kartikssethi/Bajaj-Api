import os
import uuid
import json
import traceback
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
import google.generativeai as genai
import warnings

load_dotenv()

gemini_bp = Blueprint("gemini", __name__)
CONTENT_STORE = {}  # In-memory content and embeddings: {group_id: [(chunk, embedding)]}

def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name="gemini-1.5-pro")

def get_embedding(text):
    """Get embedding from Gemini."""
    response = genai.embed_content(model="models/embedding-001", content=text)
    return response.get("embedding", [])

def cosine_similarity(a, b):
    dot = sum(i * j for i, j in zip(a, b))
    norm_a = sum(i * i for i in a) ** 0.5
    norm_b = sum(j * j for j in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-8)

def store_content_and_embeddings(file_path, group_id):
    """Store PDF chunks and embeddings for a group."""
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        chunking_strategy="by_title",
        infer_table_strategy=True,
        max_characters=1000,
        new_after_n_chars=1200,
        combine_text_under_n_chars=300,
    )
    
    stored = []
    for el in elements:
        text = str(el).strip()
        if not text:
            continue
        embedding = get_embedding(text)
        stored.append((text, embedding))

    CONTENT_STORE[group_id] = CONTENT_STORE.get(group_id, []) + stored

def fetch_relevant_context(group_id, query, top_k=5):
    """Return top-k most relevant text chunks for a query."""
    if group_id not in CONTENT_STORE:
        return []

    query_emb = get_embedding(query)
    scored = [
        (cosine_similarity(query_emb, emb), chunk)
        for chunk, emb in CONTENT_STORE[group_id]
    ]
    top = sorted(scored, reverse=True)[:top_k]
    return [chunk for _, chunk in top]

def generate_ai_response(query, context_chunks):
    """Query Gemini with relevant context."""
    warnings.filterwarnings("ignore")
    context = "\n\n".join(context_chunks)
    prompt = f"Use the context below to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"
    model = current_app.config["gemini_model"]
    response = model.generate_content(prompt)
    return response.text.strip()

@gemini_bp.route("/upload", methods=["POST"])
def upload():
    file_path = None
    try:
        group_id = request.form.get("group_id")
        if not group_id:
            return jsonify({"error": "Missing group_id"}), 400

        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No valid file provided"}), 400

        filename = secure_filename(file.filename)
        file_path = f"./tmp_{uuid.uuid4()}_{filename}"
        file.save(file_path)

        store_content_and_embeddings(file_path, group_id)
        return jsonify({"message": "✅ File uploaded and processed."}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

@gemini_bp.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        query_text = data.get("query")
        group_id = data.get("group_id")

        if not query_text or not group_id:
            return jsonify({"error": "Missing query or group_id"}), 400

        context = fetch_relevant_context(group_id, query_text)
        if not context:
            return jsonify({"error": "No relevant content found."}), 404

        answer = generate_ai_response(query_text, context)
        return jsonify({"answer": answer}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
