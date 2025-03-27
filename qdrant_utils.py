from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer  # Importer le modèle
import config
import uuid 

# Initialisation du client Qdrant
qdrant = QdrantClient(url=config.QDRANT_URL)

# Charger le modèle de vectorisation
model = SentenceTransformer("all-MiniLM-L6-v2")  # Un modèle léger et efficace

# Création de la collection (si elle n'existe pas encore)
def create_collection():
    qdrant.recreate_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Correction ici
    )
    print(f"✅ Collection '{config.COLLECTION_NAME}' créée avec succès !")

# Vérifier si la collection existe
def check_collection():
    collections = qdrant.get_collections()
    return config.COLLECTION_NAME in [c.name for c in collections.collections]

# Fonction pour stocker un fichier vectorisé
def store_vectorized_text(file_name, text):
    """Vectorise le texte et le stocke dans Qdrant."""
    try:
        print(f"🛠️ Texte brut extrait : {text[:500]}...")  # Affiche les 500 premiers caractères
        
        vector = model.encode(text).tolist()  # Convertir en liste pour Qdrant
        
        print(f"🛠️ Vecteur généré (extrait) : {vector[:5]}")  # Vérifie le début du vecteur
        
        # Création d'un point à indexer dans Qdrant
        point = PointStruct(
        id=str(uuid.uuid4()),  # Génère un UUID unique au lieu d'un hash négatif
        vector=vector,
        payload={"file_name": file_name, "text": text}
        )
        
        # Ajouter le point à la collection dans Qdrant
        qdrant.upsert(
            collection_name=config.COLLECTION_NAME,
            points=[point]
        )
        print(f"✅ Vecteur du fichier {file_name} stocké avec succès !")
    
    except Exception as e:
        print(f"❌ Erreur lors de la vectorisation : {e}")
