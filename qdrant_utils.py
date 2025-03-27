from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import config

# Initialisation du client Qdrant
qdrant = QdrantClient(url=config.QDRANT_URL)

# Création de la collection (si elle n'existe pas encore)
def create_collection():
    qdrant.recreate_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Ajout du paramètre obligatoire
    )
    print(f"✅ Collection '{config.COLLECTION_NAME}' créée avec succès !")

# Vérifier si la collection existe
def check_collection():
    collections = qdrant.get_collections()
    return config.COLLECTION_NAME in [c.name for c in collections.collections]
