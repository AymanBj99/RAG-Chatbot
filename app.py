from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import PyPDF2
import os
from dotenv import load_dotenv  # Charger .env

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

# Configuration CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000")
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

# Initialisation du modèle d'embedding
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Configuration du dossier d'upload
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Connexion à Qdrant
client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))

COLLECTION_NAME = "resumes"
try:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
except Exception as e:
    print(f"Collection already exists or error: {e}")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() or "" for page in pdf_reader.pages)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        text = extract_text_from_pdf(filepath)
        if not text.strip():
            return jsonify({"error": "No text extracted from PDF"}), 400

        embedding = embedding_model.encode(text).tolist()
        resume_id = hash(file.filename) % (10**6)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(id=resume_id, vector=embedding, payload={"text": text})]
        )

        return jsonify({"message": "Resume stored successfully", "text": text})

    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
