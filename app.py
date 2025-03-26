from flask import Flask, request, jsonify
from minio import Minio
import os

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

app = Flask(__name__)

# Configuration de MinIO avec les variables d'environnement
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Désactiver SSL pour un usage local
)

@app.route("/upload", methods=["POST"])
def upload_cv():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier trouvé"}), 400
    
    file = request.files["file"]
    file_name = file.filename
    
    try:
        # Upload vers MinIO
        minio_client.put_object(
            BUCKET_NAME,
            file_name,
            file,
            length=-1,
            part_size=10*1024*1024,
            content_type="application/pdf"
        )
        return jsonify({"message": f"Fichier {file_name} uploadé avec succès"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
