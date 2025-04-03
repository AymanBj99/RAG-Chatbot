from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from utils import extract_text_from_pdf
from qdrant_setup import create_collection

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle d'embedding
embedder = SentenceTransformer("all-mpnet-base-v2")

# Connexion à Qdrant
client = QdrantClient("localhost", port=6333)

@app.route("/api/hello")
def hello():
    return {"message": "Hello from Flask!"}

@app.route("/")
def home():
    return "Bienvenue sur l'API de filtrage de CVs !"

@app.route("/upload", methods=["POST"])
def upload_cv():
    """Upload un CV et stocke son embedding dans Qdrant"""
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Fichier invalide"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extraction du texte
    text = extract_text_from_pdf(file_path)
    embedding = embedder.encode(text).tolist()

    # Stockage dans Qdrant
    client.upsert(
        collection_name="cvs",
        points=[PointStruct(id=hash(file.filename), vector=embedding, payload={"filename": file.filename, "text": text})]
    )

    return jsonify({"message": "CV enregistré avec succès"}), 200

@app.route("/search", methods=["POST"])
def search_cv():
    """Recherche les CVs les plus pertinents en fonction d'une requête"""
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Requête vide"}), 400

    query_vector = embedder.encode(query).tolist()
    search_results = client.search(collection_name="cvs", query_vector=query_vector, limit=5)

    results = [{"filename": res.payload["filename"], "text": res.payload["text"]} for res in search_results]
    return jsonify(results), 200

@app.route("/chat", methods=["POST"])
def chat():
    """Chatbot qui répond en utilisant RAG et Ollama"""
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Requête vide"}), 400

    query_vector = embedder.encode(query).tolist()
    search_results = client.search(collection_name="cvs", query_vector=query_vector, limit=3)

    context = "\n\n".join([res.payload["text"] for res in search_results])
    prompt = f"Contexte:\n{context}\n\nQuestion: {query}\nRéponse:"

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    
    return jsonify({"response": response['message']}), 200

if __name__ == "__main__":
    create_collection()
    print("Collection 'cvs' créée avec succès !")
    app.run(debug=True)
