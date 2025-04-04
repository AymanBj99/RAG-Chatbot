from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Connexion à Qdrant (assure-toi que Qdrant est lancé)
client = QdrantClient("localhost", port=6333)

# Création de la collection pour stocker les CVs
def create_collection():
    client.recreate_collection(
        collection_name="cvs",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

if __name__ == "__main__":
    create_collection()
    print("Collection 'cvs' créée avec succès !")


    
