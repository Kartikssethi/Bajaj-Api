import os
import uuid
import json
import traceback
import warnings
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from unstructured.partition.pdf import partition_pdf
from supabase import create_client
import google.generativeai as genai
import pytesseract  # OCR tool
import PIL

# Load environment variables
load_dotenv()
ocr_agent = pytesseract.image_to_string 
gemini_bp = Blueprint("gemini", __name__)

def setup_gemini():
    """Sets up Google Gemini API. Returns a generative model."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
    
    # Ensure the GEMINI_API_KEY is set properly
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable must be set.")
    
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(model_name="gemini-2.5-pro")

def get_embedding(text):
    """Generates an embedding for the given text using the Gemini API."""
    response = genai.embed_content(model="models/embedding-001", content=text)
    return response.get("embedding", [])

def store_content_and_embeddings(file_path):
    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",  # Best for layout-aware documents
            chunking_strategy="by_title",  # Logical sectioning
            infer_table_strategy=True,  # Extracts tables as structured data
            max_characters=1000,  # Ideal chunk size for embeddings
            new_after_n_chars=1200,  # Slightly tighter wrap to avoid overlap
            combine_text_under_n_chars=300,  # Merges tiny fragments for context
        )
        return elements  # or store to DB if needed
    except Exception as e:
        print(f"Error in store_content_and_embeddings: {e}")
        raise




def make_prompt(query, context):
    """Creates a prompt for Gemini using context."""
    return f"Based on the following information, answer the question:\n\n{context}\n\nQuestion: {query}"

def generate_ai_response(user_query, context_text):
    """Generates an AI answer and saves it to the database."""
    model = current_app.config["gemini_model"]
    warnings.filterwarnings("ignore")

    prompt = make_prompt(user_query, context_text)
    response = model.generate_content(prompt)
    answer_text = response.text.strip()

    ai_user_id = user_id
    message_id = str(uuid.uuid4())


    return answer_text

@gemini_bp.route("/upload", methods=["POST"])
def upload_file():
    """Handles file uploads, extracts content, and stores embeddings."""
    supabase = current_app.config["supabase_client"]
    file_path = None

    try:
        group_id = request.form.get("group_id")
        sender_id = request.form.get("sender_id")
        sender_role = request.form.get("sender_role")

        if not all([group_id, sender_id, sender_role]):
            return jsonify({"error": "Missing group_id, sender_id, or sender_role"}), 400

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        file_path = f"./tmp_{filename}"
        file.save(file_path)

        store_content_and_embeddings(file_path, supabase, group_id, sender_id, sender_role)

        return jsonify({"message": "✅ Content and embeddings uploaded successfully!"}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

@gemini_bp.route("/query", methods=["POST"])
def ask_question():
    """Handles user queries and returns answers using Gemini or teacher context."""
    supabase = current_app.config["supabase_client"]

    try:
        data = request.get_json()
        user_query = data.get("query")
        group_id = data.get("group_id")
        user_id = data.get("user_id")

        if not user_query:
            return jsonify({"error": "Missing 'query'"}), 400

        # Return existing answer if already present
        existing = supabase.table("answers").select("*").eq("query", user_query).execute()
        if existing.data:
            return jsonify({"msg": "Answer already exists", "answer": existing.data[0]["answers"]}), 200

        # Check if teacher is available
        group_info = supabase.table("groups").select("teacher_id").eq("id", group_id).execute().data
        teacher_id = group_info[0]["teacher_id"] if group_info else None

        is_teacher_available = True
        if teacher_id:
            teacher_info = supabase.table("teachers").select("is_available").eq("id", teacher_id).execute().data
            is_teacher_available = teacher_info[0]["is_available"] if teacher_info else True

        # Fetch context
        context_data = fetch_relevant_content(supabase, group_id)
        context_text = "\n".join(context_data)

        # Generate and return AI answer
        ai_response = generate_ai_response(user_query, context_text, supabase, group_id, user_id)
        return jsonify({"answer": ai_response})
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# OCR function for extracting text from images
def extract_text_from_image(file_path):
    """Extracts text from an image using OCR."""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)  # Use the OCR agent here
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None