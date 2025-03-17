from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import PyPDF2
import os

app = Flask(__name__)
CORS(app)

client = QdrantClient(host="localhost", port=6333)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "resumes"
try:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
except Exception as e:
    print(f"Collection already exists or error: {e}")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    app.run(debug=True)