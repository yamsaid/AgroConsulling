import json
import os
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path
import numpy as np
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Pipeline optimisé pour créer et stocker des embeddings dans ChromaDB.
    Résout le problème de limite de capacité (5461 documents/batch) en utilisant
    l'insertion incrémentale avec gestion de la mémoire.
    """
    
    # Constantes ChromaDB
    CHROMADB_MAX_BATCH_SIZE = 5461  # Limite stricte de ChromaDB
    SAFE_BATCH_SIZE = 100  # Taille de lot sécurisée pour éviter les problèmes
    EMBEDDING_BATCH_SIZE = 32  # Taille optimale pour l'embedding GPU/CPU
    
    def __init__(
        self, 
        corpus_path: str, 
        chroma_db_path: str = "./chroma_db",
        collection_name: str = "agriculture_burkina",
        model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    ):
        """
        Initialise le pipeline d'embedding.
        
        Args:
            corpus_path: Chemin vers le fichier corpus.json
            chroma_db_path: Chemin de la base de données ChromaDB
            collection_name: Nom de la collection ChromaDB
            model_name: Nom du modèle SentenceTransformer
        """
        self.corpus_path = Path(corpus_path)
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        self.model_name = model_name
        
        self.embedding_model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        
        # Statistiques
        self.stats = {
            'total_chunks': 0,
            'embeddings_created': 0,
            'documents_stored': 0,
            'start_time': None,
            'end_time': None
        }
        
    def initialize_embedding_model(self) -> None:
        """
        Initialise le modèle d'embedding multilingue optimisé pour le français.
        Utilise paraphrase-multilingual-MiniLM-L12-v2 qui supporte 50+ langues.
        """
        logger.info("="*60)
        logger.info("INITIALISATION DU MODELE D'EMBEDDING")
        logger.info("="*60)
        logger.info(f"Modele: {self.model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            # Alias pour compatibilité API
            self.model = self.embedding_model
            
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            max_seq_length = self.embedding_model.max_seq_length
            
            logger.info("Modele charge avec succes")
            logger.info(f"  - Dimension des embeddings: {embedding_dim}")
            logger.info(f"  - Longueur max sequence: {max_seq_length} tokens")
            logger.info(f"  - Device: {self.embedding_model.device}")
            
        except Exception as e:
            logger.error(f"ERREUR lors du chargement du modele: {e}")
            raise
    
    def initialize_chroma(self, reset: bool = False) -> None:
        """
        Initialise la base de données vectorielle ChromaDB.
        
        Args:
            reset: Si True, supprime et recrée la collection
        """
        logger.info("="*60)
        logger.info("INITIALISATION DE CHROMADB")
        logger.info("="*60)
        logger.info(f"Chemin: {self.chroma_db_path}")
        logger.info(f"Collection: {self.collection_name}")
        
        try:
            # Créer le dossier si nécessaire
            self.chroma_db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialiser le client ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Gestion de la collection
            if reset:
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                    logger.info(f"Collection '{self.collection_name}' supprimee")
                except Exception:
                    pass  # La collection n'existe pas
            
            # Créer ou récupérer la collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Base de connaissances agricoles du Burkina Faso",
                    "created_at": datetime.now().isoformat(),
                    "model": self.model_name
                }
            )
            
            existing_count = self.collection.count()
            logger.info(f"Collection initialisee - Documents existants: {existing_count}")
            
        except Exception as e:
            logger.error(f"ERREUR lors de l'initialisation de ChromaDB: {e}")
            raise
    
    def load_corpus(self) -> List[Dict[str, Any]]:
        """
        Charge le corpus depuis le fichier JSON.
        
        Returns:
            Liste des chunks du corpus
        """
        logger.info("="*60)
        logger.info("CHARGEMENT DU CORPUS")
        logger.info("="*60)
        logger.info(f"Fichier: {self.corpus_path}")
        
        try:
            if not self.corpus_path.exists():
                raise FileNotFoundError(f"Fichier corpus introuvable: {self.corpus_path}")
            
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            
            self.stats['total_chunks'] = len(corpus)
            
            logger.info(f"Corpus charge avec succes")
            logger.info(f"  - Total chunks: {len(corpus)}")
            logger.info(f"  - Taille fichier: {self.corpus_path.stat().st_size / 1024:.2f} KB")
            
            # Validation basique
            if not corpus:
                raise ValueError("Le corpus est vide")
            
            # Vérifier la structure du premier chunk
            required_fields = ['text', 'chunk_id', 'source']
            missing_fields = [f for f in required_fields if f not in corpus[0]]
            if missing_fields:
                logger.warning(f"Champs manquants dans le corpus: {missing_fields}")
            
            return corpus
            
        except Exception as e:
            logger.error(f"ERREUR lors du chargement du corpus: {e}")
            raise
    
    def _prepare_batch_data(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> tuple[List[str], List[Dict], List[str]]:
        """
        Prépare les données d'un batch pour insertion dans ChromaDB.
        
        Args:
            chunks: Liste de chunks à traiter
            
        Returns:
            Tuple (documents, metadatas, ids)
        """
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Extraction du texte
            documents.append(chunk.get('text', ''))
            
            # Métadonnées (uniquement types supportés: str, int, float, bool)
            metadata = {
                'source': str(chunk.get('source', 'unknown')),
                'chunk_id': str(chunk.get('chunk_id', '')),
                'chunk_index': int(chunk.get('chunk_index', 0)),
                'total_chunks': int(chunk.get('total_chunks', 1)),
                'extraction_method': str(chunk.get('extraction_method', 'unknown'))
            }
            
            # Ajouter source_url si présent
            if 'source_url' in chunk:
                metadata['source_url'] = str(chunk['source_url'])
            
            metadatas.append(metadata)
            ids.append(chunk.get('chunk_id', f'chunk_{len(ids)}'))
        
        return documents, metadatas, ids
    
    def _create_embeddings_batch(self, documents: List[str]) -> List[List[float]]:
        """
        Crée les embeddings pour un batch de documents.
        Optimisé pour éviter les problèmes de mémoire.
        
        Args:
            documents: Liste de textes à encoder
            
        Returns:
            Liste d'embeddings
        """
        # Traiter par sous-lots pour optimiser la mémoire
        all_embeddings = []
        
        for i in range(0, len(documents), self.EMBEDDING_BATCH_SIZE):
            batch = documents[i:i + self.EMBEDDING_BATCH_SIZE]
            embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
    
    def create_embeddings_incremental(self, force_recreate: bool = False) -> None:
        """
        Crée les embeddings et les stocke dans ChromaDB de manière incrémentale.
        Cette méthode résout le problème de limite ChromaDB (5461 docs/batch).
        
        STRATÉGIE:
        1. Traiter le corpus par petits lots (SAFE_BATCH_SIZE)
        2. Créer les embeddings par micro-lots (EMBEDDING_BATCH_SIZE)
        3. Insérer immédiatement dans ChromaDB
        4. Libérer la mémoire après chaque insertion
        
        Args:
            force_recreate: Si True, supprime et recrée toute la collection
        """
        self.stats['start_time'] = datetime.now()
        
        # Charger le corpus
        corpus = self.load_corpus()
        
        # Réinitialiser la collection si demandé
        if force_recreate:
            logger.info("Recreation de la collection demandee")
            self.initialize_chroma(reset=True)
        else:
            existing_count = self.collection.count()
            if existing_count > 0:
                logger.warning(f"La collection contient deja {existing_count} documents")
                response = input("Continuer et ajouter de nouveaux documents ? (o/n): ")
                if response.lower() != 'o':
                    logger.info("Operation annulee par l'utilisateur")
                    return
        
        logger.info("="*60)
        logger.info("CREATION ET STOCKAGE DES EMBEDDINGS")
        logger.info("="*60)
        logger.info(f"Total chunks a traiter: {len(corpus)}")
        logger.info(f"Taille des lots: {self.SAFE_BATCH_SIZE}")
        logger.info(f"Taille micro-lots embedding: {self.EMBEDDING_BATCH_SIZE}")
        logger.info("="*60)
        
        # Compteurs pour statistiques
        total_processed = 0
        total_batches = (len(corpus) + self.SAFE_BATCH_SIZE - 1) // self.SAFE_BATCH_SIZE
        
        # Traitement par lots sécurisés
        with tqdm(total=len(corpus), desc="Progression totale", unit="chunks") as pbar:
            for batch_idx in range(0, len(corpus), self.SAFE_BATCH_SIZE):
                batch_end = min(batch_idx + self.SAFE_BATCH_SIZE, len(corpus))
                batch_chunks = corpus[batch_idx:batch_end]
                
                try:
                    # Préparer les données du batch
                    documents, metadatas, ids = self._prepare_batch_data(batch_chunks)
                    
                    # Créer les embeddings
                    embeddings = self._create_embeddings_batch(documents)
                    self.stats['embeddings_created'] += len(embeddings)
                    
                    # Insérer dans ChromaDB immédiatement
                    self.collection.add(
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    
                    self.stats['documents_stored'] += len(documents)
                    total_processed += len(batch_chunks)
                    
                    # Mise à jour de la barre de progression
                    pbar.update(len(batch_chunks))
                    pbar.set_postfix({
                        'batch': f"{(batch_idx // self.SAFE_BATCH_SIZE) + 1}/{total_batches}",
                        'stored': self.stats['documents_stored']
                    })
                    
                    # Log détaillé tous les 10 lots
                    if (batch_idx // self.SAFE_BATCH_SIZE + 1) % 10 == 0:
                        logger.info(
                            f"Progression: {total_processed}/{len(corpus)} chunks "
                            f"({total_processed/len(corpus)*100:.1f}%)"
                        )
                    
                except Exception as e:
                    logger.error(
                        f"ERREUR lors du traitement du lot {batch_idx}-{batch_end}: {e}"
                    )
                    # Continuer avec le prochain lot
                    continue
        
        self.stats['end_time'] = datetime.now()
        self._print_final_stats()
    
    def _print_final_stats(self) -> None:
        """Affiche les statistiques finales du traitement"""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("STATISTIQUES FINALES")
        logger.info("="*60)
        logger.info(f"Total chunks corpus: {self.stats['total_chunks']}")
        logger.info(f"Embeddings crees: {self.stats['embeddings_created']}")
        logger.info(f"Documents stockes: {self.stats['documents_stored']}")
        logger.info(f"Duree totale: {duration:.2f} secondes")
        logger.info(f"Vitesse: {self.stats['embeddings_created']/duration:.2f} embeddings/sec")
        logger.info(f"Base de donnees: {self.chroma_db_path}")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Taille collection: {self.collection.count()} documents")
        logger.info("="*60)
    
    def test_retrieval(
        self, 
        query: str = "Quel engrais pour le mil ?", 
        n_results: int = 3
    ) -> List[Dict]:
        """
        Teste le système de retrieval avec une requête.
        
        Args:
            query: Question à rechercher
            n_results: Nombre de résultats à retourner
            
        Returns:
            Liste des résultats trouvés
        """
        logger.info(f"Test de retrieval: '{query}'")
        
        if not self.embedding_model or not self.collection:
            raise ValueError("Le pipeline n'est pas initialise")
        
        # Encoder la requête
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Recherche dans ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Affichage des résultats
        print("\n" + "="*80)
        print(f"RESULTATS POUR: '{query}'")
        print("="*80)
        
        search_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            result = {
                'rank': i + 1,
                'document': doc,
                'metadata': metadata,
                'distance': distance,
                'similarity': 1 - distance  # Convertir distance en similarité
            }
            search_results.append(result)
            
            print(f"\n[Resultat {i+1}]")
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Chunk: {metadata.get('chunk_index', 0)}/{metadata.get('total_chunks', 0)}")
            print(f"Similarite: {result['similarity']:.4f}")
            print(f"Texte: {doc[:250]}...")
            print("-"*80)
        
        return search_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Récupère les informations sur la collection ChromaDB.
        
        Returns:
            Dictionnaire avec les informations de la collection
        """
        if not self.collection:
            raise ValueError("La collection n'est pas initialisee")
        
        info = {
            'name': self.collection_name,
            'count': self.collection.count(),
            'metadata': self.collection.metadata,
        }
        
        # Affichage formaté
        print("\n" + "="*60)
        print("INFORMATIONS COLLECTION")
        print("="*60)
        print(f"Nom: {info['name']}")
        print(f"Documents: {info['count']}")
        print(f"Metadonnees: {json.dumps(info['metadata'], indent=2)}")
        print("="*60)
        
        return info


def main():
    """Fonction principale du pipeline"""
    
    # Configuration des chemins
    CORPUS_PATH = "./data/corpus.json"
    CHROMA_DB_PATH = "./data/chroma_db"
    COLLECTION_NAME = "agriculture_burkina"
    
    # Initialisation du pipeline
    logger.info("Demarrage du pipeline d'embedding")
    
    pipeline = EmbeddingPipeline(
        corpus_path=CORPUS_PATH,
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Étape 1: Initialiser le modèle
    pipeline.initialize_embedding_model()
    
    # Étape 2: Initialiser ChromaDB
    pipeline.initialize_chroma(reset=False)  # Mettre True pour recréer
    
    # Étape 3: Créer et stocker les embeddings de manière incrémentale
    pipeline.create_embeddings_incremental(force_recreate=False)
    
    # Étape 4: Afficher les infos de la collection
    pipeline.get_collection_info()
    
    # Étape 5: Tests de retrieval
    test_queries = [
        "Quel engrais pour le mil ?",
        "Techniques de conservation des eaux et des sols",
        "Maladies du maïs au Burkina Faso",
        "Comment lutter contre la sécheresse ?",
        "Variétés de sorgho résistantes"
    ]
    
    print("\n" + "="*80)
    print("TESTS DE RETRIEVAL")
    print("="*80)
    
    for query in test_queries:
        pipeline.test_retrieval(query, n_results=3)
        print("\n")


if __name__ == "__main__":
    main()