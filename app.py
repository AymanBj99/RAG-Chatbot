from flask import Flask, request, jsonify
from flask_cors import CORS
from minio_utils import upload_file, create_bucket
from qdrant_utils import create_collection, check_collection
from pdf_utils import extract_text_from_pdf


import os

app = Flask(__name__)
CORS(app)  # Autorise React à appeler Flask

# Créer le bucket au démarrage
create_bucket()

@app.route("/upload", methods=["POST"])
def upload_cv():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier trouvé"}), 400

    file = request.files["file"]
    file_name = file.filename

    try:
        # Upload vers MinIO
        upload_file(file, file_name)

        return jsonify({"message": f"Fichier {file_name} uploadé avec succès"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/extract-text", methods=["POST"])
def extract_text():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier trouvé"}), 400

    file = request.files["file"]
    
    try:
        text = extract_text_from_pdf(file)
        return jsonify({"text": text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Créer la collection au démarrage si elle n'existe pas
if not check_collection():
    create_collection()

if __name__ == "__main__":
    app.run(debug=True)
