import json
import os
import logging
from typing import Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(self, corpus_path, chroma_db_path="./chroma_db"):
        self.corpus_path = corpus_path
        self.chroma_db_path = chroma_db_path
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
    def initialize_embedding_model(self):
        """
        Initialise le modèle d'embedding multilingue
        """
        logger.info("Chargement du modèle d'embedding...")
        try:
            # Modèle multilingue optimisé pour le français
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
            logger.info("✓ Modèle d'embedding chargé avec succès")
            logger.info(f"  - Dimension des embeddings: {self.embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    def initialize_chroma(self):
        """
        Initialise la base de données vectorielle Chroma
        """
        logger.info("Initialisation de la base de données Chroma...")
        try:
            # Configuration de Chroma
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Créer ou récupérer la collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="agriculture_burkina",
                metadata={"description": "Base de connaissances agricoles du Burkina Faso"}
            )
            logger.info("✓ Base de données Chroma initialisée")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de Chroma: {e}")
            raise
    
    def load_corpus(self):
        """
        Charge le corpus nettoyé
        """
        logger.info(f"Chargement du corpus depuis {self.corpus_path}...")
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            logger.info(f"✓ Corpus chargé: {len(corpus)} chunks")
            return corpus
        except Exception as e:
            logger.error(f"Erreur lors du chargement du corpus: {e}")
            raise
    
    def create_embeddings(self):
        """
        Crée les embeddings et les stocke dans Chroma
        """
        corpus = self.load_corpus()
        
        # Vérifier si la collection est déjà peuplée
        if self.collection.count() > 0:
            logger.warning("La collection contient déjà des données. Vidage...")
            self.chroma_client.delete_collection("agriculture_burkina")
            self.initialize_chroma()
        
        # Préparer les données
        documents = []
        metadatas = []
        ids = []
        
        for chunk in tqdm[Any](corpus, desc="Préparation des données"):
            documents.append(chunk['text'])
            metadatas.append({
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'chunk_index': chunk['chunk_index'],
                'total_chunks': chunk['total_chunks']
            })
            ids.append(chunk['chunk_id'])
        
        # Créer les embeddings par lots pour économiser la mémoire
        logger.info("Création des embeddings...")
        batch_size = 100
        all_embeddings = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Génération des embeddings"):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_docs)
            all_embeddings.extend(batch_embeddings)
        
        # Convertir en listes Python pour Chroma
        embeddings_list = [embedding.tolist() for embedding in all_embeddings]
        
        # Ajouter à Chroma par lots (taille max: 5461 selon ChromaDB)
        logger.info("Ajout des données à la base vectorielle...")
        chroma_batch_size = 5000  # En dessous de la limite de 5461
        
        total_added = 0
        for i in tqdm(range(0, len(documents), chroma_batch_size), desc="Ajout à ChromaDB"):
            batch_end = min(i + chroma_batch_size, len(documents))
            
            self.collection.add(
                embeddings=embeddings_list[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            total_added += batch_end - i
        
        logger.info(f"✓ {total_added} documents vectorisés et stockés")
        logger.info(f"✓ Base de données sauvegardée dans: {self.chroma_db_path}")
    
    def test_retrieval(self, query="Quel engrais pour le mil ?", n_results=3):
        """
        Teste le système de retrieval
        """
        logger.info(f"Test de retrieval: '{query}'")
        
        # Embedding de la requête
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Recherche dans Chroma
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Afficher les résultats
        print(f"\n🔍 RÉSULTATS POUR: '{query}'\n")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"📄 Résultat {i+1} (Source: {metadata['source']}):")
            print(f"   {doc[:200]}...")
            print(f"   Distance: {results['distances'][0][i]:.4f}")
            print("-" * 80)

def main():
    # Chemins
    corpus_path = "./data/corpus_tes.json"
    chroma_db_path = "./data/chroma_db"
    
    # Initialisation du pipeline
    pipeline = EmbeddingPipeline(corpus_path, chroma_db_path)
    
    # Exécution
    pipeline.initialize_embedding_model()
    pipeline.initialize_chroma()
    pipeline.create_embeddings()
    
    # Test
    test_queries = [
        "Quel engrais pour le mil ?",
        "Techniques de conservation des eaux",
        "Maladies du maïs au Burkina"
    ]
    
    for query in test_queries:
        pipeline.test_retrieval(query)

if __name__ == "__main__":
    main()