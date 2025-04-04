from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
import os
import zipfile
import io
from dotenv import load_dotenv  # 👈 Charger dotenv
from minio import Minio
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from utils import extract_text_from_pdf
from qdrant_setup import create_collection

# 🔹 Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔹 Configuration MinIO (via .env)
MINIO_CLIENT = Minio(
    os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)
BUCKET_NAME = os.getenv("MINIO_BUCKET")

# Vérifier si le bucket MinIO existe, sinon le créer
if not MINIO_CLIENT.bucket_exists(BUCKET_NAME):
    MINIO_CLIENT.make_bucket(BUCKET_NAME)

# 🔹 Charger le modèle d'embedding
embedder = SentenceTransformer("all-mpnet-base-v2")

# 🔹 Connexion à Qdrant (via .env)
client = QdrantClient(
    os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT"))
)

@app.route("/api/hello")
def hello():
    return {"message": "Hello from Flask!"}

@app.route("/")
def home():
    return "Bienvenue sur l'API de filtrage de CVs !"

@app.route("/upload-folder", methods=["POST"])
def upload_folder():
    """Upload un dossier ZIP contenant des PDFs, stocke dans MinIO, et indexe dans Qdrant"""
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename.endswith(".zip"):
        return jsonify({"error": "Veuillez uploader un fichier ZIP"}), 400

    # 🔹 Extraction des PDFs du ZIP
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
        extracted_files = [name for name in zip_ref.namelist() if name.endswith('.pdf')]
        if not extracted_files:
            return jsonify({"error": "Le ZIP ne contient pas de fichiers PDF"}), 400

        for file_name in extracted_files:
            with zip_ref.open(file_name) as file_data:
                file_content = file_data.read()
                minio_path = f"pdfs/{file_name}"
                
                # 🔹 Upload dans MinIO
                MINIO_CLIENT.put_object(
                    BUCKET_NAME, minio_path, io.BytesIO(file_content), length=len(file_content), part_size=10*1024*1024
                )

    # 🔹 Indexation des PDFs
    for file_name in extracted_files:
        process_and_index_pdf(file_name)

    return jsonify({"message": f"{len(extracted_files)} fichiers ont été traités et indexés"}), 200

def process_and_index_pdf(file_name):
    """Télécharge un PDF depuis MinIO, extrait le texte, génère les embeddings et stocke dans Qdrant."""
    minio_path = f"pdfs/{file_name}"
    response = MINIO_CLIENT.get_object(BUCKET_NAME, minio_path)

    # 🔹 Lire le PDF depuis MinIO correctement
    pdf_content = io.BytesIO(response.read())
    response.close()  # Toujours fermer après lecture

    # 🔹 Extraire le texte
    text = extract_text_from_pdf(pdf_content)

    # 🔹 Vérifier si le texte n'est pas vide
    if not text.strip():
        print(f"⚠️  Texte vide pour {file_name}, non indexé")
        return

    # 🔹 Générer l'embedding du texte
    embedding = embedder.encode(text).tolist()

    # 🔹 Stocker dans Qdrant
    client.upsert(
        collection_name="cvs",
        points=[PointStruct(id=hash(file_name), vector=embedding, payload={"filename": file_name, "text": text})]
    )

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
