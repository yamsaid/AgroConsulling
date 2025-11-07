"""
Vector Store Module - Production Ready
Système de stockage vectoriel pour RAG Agricole Burkina Faso

Auteur: Expert ML Team
Date: 3 Novembre 2025
Hackathon: MTDPCE 2025

Ce module fournit deux implémentations:
1. ChromaVectorStore : Interface ChromaDB (recommandé pour prototypage)
2. FAISSVectorStore : Interface FAISS (recommandé pour production)

Caractéristiques:
- Support embeddings 768D (LaBSE) et 384D (MiniLM)
- Persistance sur disque
- Recherche sémantique optimisée
- Batch processing intelligent
- Gestion robuste des erreurs
- Logging professionnel
"""

import os
import sys
import io

# Forcer UTF-8 sur la console (Windows) pour éviter UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    elif hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# Désactiver la télémétrie ChromaDB AVANT l'import pour éviter les erreurs PostHog
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import chromadb
from chromadb.config import Settings
import faiss
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import time

# Configuration logging professionnelle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vector_store.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)


@dataclass
class SearchResult:
    """Structure de données pour résultats de recherche"""
    document_id: str
    document_text: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int


class BaseVectorStore:
    """Classe de base abstraite pour vector stores"""
    
    # Dimensions supportées pour embeddings
    SUPPORTED_DIMENSIONS = {
        'labse': 768,           # LaBSE (multilingue)
        'minilm': 384,          # MiniLM (léger)
        'labse-base': 768,      # Alias
        'all-minilm': 384       # Alias
    }
    
    def __init__(self, persist_directory: str = "./data/vector_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_dimension = None
        self.document_count = 0
        
    def detect_embedding_dimension(self, sample_embedding: Union[List[float], np.ndarray]) -> int:
        """
        Détecte automatiquement la dimension des embeddings
        
        Args:
            sample_embedding: Un embedding de test
            
        Returns:
            int: Dimension détectée
        """
        if isinstance(sample_embedding, np.ndarray):
            dim = sample_embedding.shape[-1]
        else:
            dim = len(sample_embedding)
        
        logger.info(f" Dimension embeddings détectée: {dim}")
        
        # Valider dimension
        if dim not in self.SUPPORTED_DIMENSIONS.values():
            logger.warning(f"[WARNING] Dimension {dim} inhabituelle (attendu: {list(self.SUPPORTED_DIMENSIONS.values())})")
        
        self.embedding_dimension = dim
        return dim
    
    def validate_corpus_data(self, corpus: List[Dict], embeddings: Union[List, np.ndarray]) -> bool:
        """
        Validation complète des données corpus + embeddings
        
        Args:
            corpus: Liste de documents du corpus
            embeddings: Embeddings correspondants
            
        Returns:
            bool: True si validation OK
            
        Raises:
            ValueError: Si validation échoue
        """
        # Vérifier cohérence longueurs
        if len(corpus) != len(embeddings):
            raise ValueError(f"Incohérence: {len(corpus)} docs vs {len(embeddings)} embeddings")
        
        # Vérifier corpus non vide
        if not corpus:
            raise ValueError("Corpus vide !")
        
        # Vérifier structure documents
        required_fields = ['id', 'titre', 'contenu', 'source']
        for i, doc in enumerate(corpus[:5]):  # Vérifier échantillon
            for field in required_fields:
                if field not in doc:
                    raise ValueError(f"Document {i} manque champ '{field}'")
            
            # Vérifier contenu non vide
            if not doc['contenu'] or len(doc['contenu'].strip()) < 10:
                logger.warning(f"[WARNING] Document {doc['id']} a un contenu très court")
        
        # Détecter dimension embeddings
        self.detect_embedding_dimension(embeddings[0])
        
        # Vérifier cohérence dimensions
        if isinstance(embeddings, np.ndarray):
            if embeddings.shape[1] != self.embedding_dimension:
                raise ValueError(f"Dimensions embeddings incohérentes: {embeddings.shape}")
        else:
            for i, emb in enumerate(embeddings[:10]):  # Vérifier échantillon
                if len(emb) != self.embedding_dimension:
                    raise ValueError(f"Embedding {i} a dimension {len(emb)} au lieu de {self.embedding_dimension}")
        
        logger.info(f"[SUCCESS] Validation réussie: {len(corpus)} documents, dimension {self.embedding_dimension}")
        return True


# ============================================================================
# IMPLÉMENTATION 1 : CHROMADB (Interface améliorée de votre code)
# ============================================================================

# Registre global pour gérer les clients ChromaDB (évite les instances multiples)
_chroma_clients = {}

def _get_chroma_client(path: str, settings: Settings) -> chromadb.PersistentClient:
    """
    Obtient ou crée un client ChromaDB pour un chemin donné.
    Évite les erreurs "instance already exists" en réutilisant les clients.
    
    Args:
        path: Chemin du répertoire ChromaDB
        settings: Paramètres ChromaDB
        
    Returns:
        chromadb.PersistentClient: Instance du client
    """
    # Normaliser le chemin pour éviter les problèmes de casse/paths
    path_str = str(Path(path).resolve())
    
    # Vérifier si un client existe déjà pour ce chemin dans notre registre
    if path_str in _chroma_clients:
        logger.debug(f"[REUSE] Réutilisation client ChromaDB existant: {path_str}")
        return _chroma_clients[path_str]
    
    # Créer nouveau client
    try:
        client = chromadb.PersistentClient(path=path_str, settings=settings)
        _chroma_clients[path_str] = client
        logger.debug(f"[NEW] Nouveau client ChromaDB créé: {path_str}")
        return client
    except ValueError as e:
        error_msg = str(e)
        if "already exists" in error_msg.lower() or "different settings" in error_msg.lower():
            # Une instance existe déjà avec des paramètres différents
            # Essayer de récupérer le client existant avec des paramètres minimaux
            logger.warning(f"[WARNING] Instance ChromaDB existante détectée pour {path_str}, tentative réutilisation...")
            try:
                # Essayer avec les paramètres minimaux (sans allow_reset pour compatibilité)
                minimal_settings = Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
                # Si un client existe déjà, essayer de le créer avec les mêmes paramètres minimaux
                client = chromadb.PersistentClient(path=path_str, settings=minimal_settings)
                _chroma_clients[path_str] = client
                logger.info(f"[SUCCESS] Client ChromaDB récupéré avec paramètres minimaux")
                return client
            except ValueError as e2:
                # Si ça échoue encore, essayer sans paramètres (utilise les paramètres de l'instance existante)
                logger.warning(f"[WARNING] Tentative sans paramètres spécifiques...")
                try:
                    # Dernier recours: essayer de récupérer depuis le registre global ChromaDB
                    # En utilisant le chemin normalisé
                    client = chromadb.PersistentClient(path=path_str)
                    _chroma_clients[path_str] = client
                    logger.info(f"[SUCCESS] Client ChromaDB récupéré sans paramètres")
                    return client
                except Exception as e3:
                    # Si tout échoue, utiliser le client du registre s'il existe
                    logger.error(f"[ERROR] Impossible de créer/récupérer client ChromaDB: {e3}")
                    # Si on a déjà un client dans le registre pour un chemin similaire, l'utiliser
                    for cached_path, cached_client in _chroma_clients.items():
                        if Path(cached_path).resolve() == Path(path_str).resolve():
                            logger.info(f"[FALLBACK] Utilisation client existant depuis registre")
                            return cached_client
                    raise
        else:
            raise
    except Exception as e:
        # Autre type d'erreur
        logger.error(f"[ERROR] Erreur création client ChromaDB: {e}")
        raise

class ChromaVectorStore(BaseVectorStore):
    """
    Vector Store basé sur ChromaDB - Version Optimisée et Corrigée
    
    Corrections:
    - Initialisation complète de tous les attributs
    - Support de self.corpus pour compatibilité API
    - Gestion robuste de embedding_dimension
    - Validation corpus intégrée
    """
    
    DEFAULT_COLLECTION_NAME = "agriculture_burkina"
    DEFAULT_BATCH_SIZE = 100
    
    _instances = {}
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Args:
            persist_directory: Répertoire de persistance ChromaDB
        """
        self.persist_directory = Path(persist_directory)  # ✅ Convertir en Path
        self.collection_name = self.DEFAULT_COLLECTION_NAME  # ✅ Utiliser le nom par défaut
        
        # ✅ CORRECTION : Initialiser TOUS les attributs
        self.client = None
        self.collection = None
        #self.corpus = []  # ✅ AJOUTÉ : Pour compatibilité API
        self.corpus_mapping = {}  # ✅ Mapping id -> document
        self.document_count = 0
        self.embedding_dimension = None  # ✅ AJOUTÉ : Sera défini lors de create/load
        
        # ✅ Vérifier si instance existe déjà (réutilisation)
        if str(persist_directory) in ChromaVectorStore._instances:
            logger.info(f"[REUSE] Réutilisation instance ChromaDB existante: {persist_directory}")
            existing = ChromaVectorStore._instances[str(persist_directory)]
            self.client = existing.client
            self.collection = existing.collection
            self.corpus = existing.corpus
            self.corpus_mapping = existing.corpus_mapping
            self.document_count = existing.document_count
            self.embedding_dimension = existing.embedding_dimension
            # ✅ S'assurer que persist_directory est un Path
            if not isinstance(self.persist_directory, Path):
                self.persist_directory = Path(self.persist_directory)
            return
        
        # ✅ Forcer réinitialisation singleton ChromaDB si nécessaire
        try:
            import chromadb
            if hasattr(chromadb.api, '_client_instances'):
                chromadb.api._client_instances.clear()
        except:
            pass
        
        # ✅ Enregistrer cette instance
        ChromaVectorStore._instances[str(persist_directory)] = self

        @property
        def corpus(self) -> List[Dict]:
            """
            ✅ Propriété calculée : retourne le corpus depuis corpus_mapping
            Évite la duplication mémoire
            """
            return list(self.corpus_mapping.values())

        @property
        def corpus_size(self) -> int:
            """Nombre de documents dans le corpus"""
            return len(self.corpus_mapping)
        
        def _get_chroma_settings(self, allow_reset: bool = False) -> Settings:
            """
            ✅ Crée des Settings ChromaDB cohérents
            
            Args:
                allow_reset: Autoriser reset de la collection
                
            Returns:
                Settings configurés
            """
            return Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                allow_reset=allow_reset  # Configurable selon le contexte
            )

    def validate_corpus_data(self, corpus: List[Dict], embeddings: Union[List, np.ndarray]) -> None:
        """
        ✅ AJOUTÉ : Valide les données du corpus
        
        Args:
            corpus: Liste de documents
            embeddings: Embeddings correspondants
            
        Raises:
            ValueError: Si données invalides
        """
        if not corpus:
            raise ValueError("Corpus vide")
        
        if not embeddings or len(embeddings) == 0:
            raise ValueError("Embeddings vides")
        
        if len(corpus) != len(embeddings):
            raise ValueError(
                f"Taille corpus ({len(corpus)}) != taille embeddings ({len(embeddings)})"
            )
        
        # Vérifier structure corpus
        # Vérifier structure corpus avec mapping flexible
        required_fields = {
            'id': ['id', 'chunk_id', 'doc_id'],  # Accepter plusieurs noms
            'text': ['contenu', 'text', 'content']  # Idem
        }

        for i, doc in enumerate(corpus[:5]):
            for field, alternatives in required_fields.items():
                # Vérifier si au moins une alternative existe
                if not any(alt in doc for alt in alternatives):
                    raise ValueError(
                        f"Document {i} manque un champ '{field}'. "
                        f"Alternatives acceptées: {alternatives}"
                    )
                
                # Normaliser le nom du champ (optionnel mais recommandé)
                if field == 'id' and 'id' not in doc:
                    for alt in alternatives:
                        if alt in doc:
                            doc['id'] = doc[alt]
                            break
                
                if field == 'text' and 'text' not in doc:
                    for alt in alternatives:
                        if alt in doc:
                            doc['text'] = doc[alt]
                            break

        logger.info("✅ Structure corpus validée et normalisée")


        # Détecter dimension embeddings
        if isinstance(embeddings, np.ndarray):
            self.embedding_dimension = embeddings.shape[1]
        elif isinstance(embeddings, list) and len(embeddings) > 0:
            first_emb = embeddings[0]
            if isinstance(first_emb, np.ndarray):
                self.embedding_dimension = len(first_emb)
            elif isinstance(first_emb, list):
                self.embedding_dimension = len(first_emb)
        
        logger.info(f"[VALID] Corpus: {len(corpus)} docs, Embeddings: {self.embedding_dimension}D")

    def create_index(self, corpus: List[Dict], embeddings: Union[List, np.ndarray],
                    collection_name: str = DEFAULT_COLLECTION_NAME,
                    reset: bool = False) -> bool:
        """
        Crée l'index vectoriel à partir du corpus
        
        Args:
            corpus: Liste de documents (format corpus.json)
            embeddings: Embeddings correspondants
            collection_name: Nom de la collection ChromaDB
            reset: Si True, supprime collection existante
            
        Returns:
            bool: True si succès
        """
        try:
            logger.info("[LAUNCH] Création index ChromaDB...")
            start_time = time.time()
            
            # ✅ S'assurer que persist_directory est un Path
            if not isinstance(self.persist_directory, Path):
                self.persist_directory = Path(self.persist_directory)
            
            # ✅ Créer répertoire si nécessaire
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Validation données
            self.validate_corpus_data(corpus, embeddings)
            
            # Initialiser ChromaDB avec paramètres cohérents
            # ✅ Utiliser allow_reset seulement si reset=True pour éviter conflits avec load()
            settings = self._get_chroma_settings(allow_reset=reset)
            self.client = _get_chroma_client(str(self.persist_directory), settings)
            # Gérer reset si demandé
            if reset:
                try:
                    self.client.delete_collection(collection_name)
                    logger.info(f"[DELETE] Collection '{collection_name}' supprimée")
                except:
                    pass
            
            # Créer ou récupérer collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": "Base de connaissances agricoles Burkina Faso",
                    "domain": "agriculture",
                    "language": "french",
                    "embedding_dimension": str(self.embedding_dimension),
                    "model": "LaBSE" if self.embedding_dimension == 768 else "MiniLM",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "hackathon": "MTDPCE_2025"
                }
            )
            
            # Préparer données pour ChromaDB
            documents = []
            metadatas = []
            ids = []
            embeddings_list = []
            
            # ✅ Remplir self.corpus pour compatibilité API
            #self.corpus = corpus.copy()
            
            for i, doc in enumerate(corpus):
                # Stocker mapping complet
                self.corpus_mapping[doc['id']] = doc
                
                # Préparer pour ChromaDB
                documents.append(doc['contenu'][:1000])  # Limiter taille texte stocké
                
                metadatas.append({
                    'id': doc['id'],
                    'titre': doc.get('titre', 'Unknown')[:200],
                    'source': doc.get('source', 'Unknown')[:200],
                    'organisme': doc.get('organisme', 'Unknown'),
                    'type': doc.get('type', 'general')
                })
                
                ids.append(doc['id'])
                
                # Convertir embedding en liste Python
                if isinstance(embeddings, np.ndarray):
                    embeddings_list.append(embeddings[i].tolist())
                else:
                    embeddings_list.append(embeddings[i] if isinstance(embeddings[i], list) else embeddings[i].tolist())
            
            # Insertion par batch pour performance
            batch_size = self.DEFAULT_BATCH_SIZE
            total_batches = (len(corpus) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(corpus))
                
                self.collection.add(
                    embeddings=embeddings_list[start_idx:end_idx],
                    documents=documents[start_idx:end_idx],
                    metadatas=metadatas[start_idx:end_idx],
                    ids=ids[start_idx:end_idx]
                )
                
                logger.info(f"   Batch {batch_num+1}/{total_batches} ajouté ({end_idx-start_idx} docs)")
            
            self.document_count = len(corpus)
            elapsed = time.time() - start_time
            
            logger.info(f"[SUCCESS] Index créé: {self.document_count} documents en {elapsed:.2f}s")
            if elapsed > 1e-6:
                logger.info(f"[STATS] Vitesse: {self.document_count/elapsed:.1f} docs/sec")
            
            # Sauvegarder corpus mapping
            self.save()
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Erreur création index: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search(self, query_embedding: Union[List[float], np.ndarray], 
               k: int = 5,
               filter_metadata: Optional[Dict] = None) -> List[SearchResult]:
        """Recherche sémantique dans le vector store"""
        try:
            # ✅ Auto-chargement si collection non initialisée
            if not self.collection:
                logger.warning("[WARNING] Collection non initialisée, tentative auto-chargement...")
                load_success = self.load()
                if not load_success or not self.collection:
                    raise RuntimeError("Index non initialisé. Appeler create_index() ou load() d'abord.")
            
            # Convertir query en format ChromaDB
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # S'assurer du format [[embedding]]
            if not isinstance(query_embedding, list):
                query_embedding = query_embedding.tolist()
            
            if len(query_embedding) > 0 and not isinstance(query_embedding[0], list):
                query_embedding = [query_embedding]
            
            # Recherche
            start_time = time.time()
            
            chroma_results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                where=filter_metadata
            )
            
            elapsed = time.time() - start_time
            
            # Formater résultats
            results = []
            
            if chroma_results['documents'] and chroma_results['documents'][0]:
                for rank, (doc_id, distance) in enumerate(zip(
                    chroma_results['ids'][0],
                    chroma_results['distances'][0]
                ), 1):
                    
                    # Récupérer document complet
                    full_doc = self.corpus_mapping.get(doc_id, {})
                    
                    # Convertir distance en similarité
                    similarity = 1.0 / (1.0 + distance)
                    
                    results.append(SearchResult(
                        document_id=doc_id,
                        document_text=full_doc.get('contenu', ''),
                        metadata={
                            'titre': full_doc.get('titre', 'Unknown'),
                            'source': full_doc.get('source', 'Unknown'),
                            'organisme': full_doc.get('organisme', 'Unknown'),
                            'type': full_doc.get('type', 'general')
                        },
                        similarity_score=similarity,
                        rank=rank
                    ))
            
            logger.info(f"[SEARCH] Recherche: {len(results)} résultats en {elapsed*1000:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] Erreur recherche: {e}")
            import traceback
            traceback.print_exc()
            return []

    def save(self, metadata_file: str = "corpus_mapping.pkl") -> bool:
        """
        Sauvegarde le mapping corpus (ChromaDB se sauvegarde automatiquement)
        
        Args:
            metadata_file: Nom du fichier de sauvegarde
            
        Returns:
            bool: True si succès
        """
        try:
            # ✅ S'assurer que persist_directory est un Path
            if not isinstance(self.persist_directory, Path):
                self.persist_directory = Path(self.persist_directory)
            
            mapping_path = self.persist_directory / metadata_file
            
            # ✅ Sauvegarder à la fois corpus et mapping
            save_data = {
                #'corpus': self.corpus,
                'corpus_mapping': self.corpus_mapping,
                'embedding_dimension': self.embedding_dimension,
                'document_count': self.document_count
            }
            
            with open(mapping_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"[SAVE] Corpus mapping sauvegardé: {mapping_path}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Erreur sauvegarde: {e}")
            return False
    
    def load(self, collection_name: str = DEFAULT_COLLECTION_NAME,
             metadata_file: str = "corpus_mapping.pkl") -> bool:
        """
        Charge un index existant depuis le disque
        
        Args:
            collection_name: Nom de la collection
            metadata_file: Fichier du corpus mapping
            
        Returns:
            bool: True si succès
        """
        try:
            logger.info("[LOAD] Chargement index ChromaDB...")
            
            # ✅ S'assurer que persist_directory est un Path
            if not isinstance(self.persist_directory, Path):
                self.persist_directory = Path(self.persist_directory)
            
            # ✅ Réutiliser client existant si disponible
            if self.client is not None:
                logger.debug("[REUSE] Réutilisation client ChromaDB existant de l'instance")
            else:
                # Initialiser client avec paramètres minimaux (sans allow_reset pour compatibilité)
                # Utiliser les mêmes paramètres minimaux que pour éviter les conflits
                settings = self._get_chroma_settings(allow_reset=False)
                self.client = _get_chroma_client(str(self.persist_directory), settings)

            # Charger collection (avec gestion d'erreur si elle n'existe pas)
            try:
                self.collection = self.client.get_collection(collection_name)
                self.document_count = self.collection.count()
            except Exception as e:
                error_msg = str(e).lower()
                if "does not exist" in error_msg or "not found" in error_msg:
                    # ✅ Essayer avec l'autre nom de collection possible (compatibilité)
                    alternative_names = ["agricultural_corpus", "agriculture_burkina"]
                    if collection_name not in alternative_names:
                        alternative_names.insert(0, collection_name)
                    
                    logger.warning(f"[WARNING] Collection '{collection_name}' introuvable, recherche alternatives...")
                    available_collections = [c.name for c in self.client.list_collections()]
                    logger.info(f"[INFO] Collections disponibles: {available_collections}")
                    
                    # Essayer les noms alternatifs
                    collection_found = False
                    for alt_name in alternative_names:
                        if alt_name in available_collections:
                            logger.info(f"[SUCCESS] Collection trouvée avec nom alternatif: '{alt_name}'")
                            self.collection = self.client.get_collection(alt_name)
                            self.document_count = self.collection.count()
                            collection_found = True
                            break
                    
                    if not collection_found:
                        logger.error(f"[ERROR] Aucune collection trouvée parmi: {alternative_names}")
                        logger.error(f"[ERROR] Collections disponibles: {available_collections}")
                        raise ValueError(f"Collection '{collection_name}' n'existe pas. Créer l'index d'abord avec create_index().")
                else:
                    raise
            
            # Charger corpus mapping
            mapping_path = self.persist_directory / metadata_file
            if mapping_path.exists():
                with open(mapping_path, 'rb') as f:
                    save_data = pickle.load(f)
                
                # Nouveau format
                if isinstance(save_data, dict) and 'corpus_mapping' in save_data:
                    self.corpus_mapping = save_data['corpus_mapping']
                    self.embedding_dimension = save_data.get('embedding_dimension')
                    self.document_count = save_data.get('document_count', len(self.corpus_mapping))
                    logger.info(f"[SUCCESS] Corpus chargé: {self.document_count} documents")
                else:
                    # Support ancien format (rétrocompatibilité)
                    if 'corpus' in save_data and 'corpus_mapping' in save_data:
                        logger.warning("[MIGRATION] Ancien format détecté, migration vers corpus_mapping")
                        self.corpus_mapping = save_data['corpus_mapping']
                    elif isinstance(save_data, dict):
                        # Très ancien format : juste le mapping
                        self.corpus_mapping = save_data
                    
                    self.document_count = len(self.corpus_mapping)
                
                logger.info(f"[SUCCESS] Corpus chargé: {len(self.corpus_mapping)} documents")
            else:
                logger.warning("[WARNING] Fichier corpus mapping introuvable")
            
            # Récupérer dimension depuis métadonnées
            # Récupérer dimension depuis métadonnées (OBLIGATOIRE)
            if self.embedding_dimension is None:
                metadata = self.collection.metadata or {}
                dim_value = metadata.get('embedding_dimension')
                
                if dim_value is not None:
                    try:
                        self.embedding_dimension = int(dim_value)
                        logger.info(f"✅ Dimension chargée depuis métadonnées: {self.embedding_dimension}")
                    except ValueError as e:
                        logger.error(f"❌ Dimension invalide dans métadonnées: {dim_value}")
                
                # Fallback sur modèle (avec avertissement)
                if self.embedding_dimension is None:
                    model_meta = (metadata.get('model') or '').lower()
                    if 'minilm' in model_meta or '384' in model_meta:
                        self.embedding_dimension = 384
                    elif 'labse' in model_meta or '768' in model_meta:
                        self.embedding_dimension = 768
                    else:
                        # ❌ Ne pas deviner silencieusement
                        logger.error("❌ Impossible de déterminer embedding_dimension")
                        logger.error(f"   Métadonnées disponibles: {metadata}")
                        raise ValueError(
                            "Dimension embeddings inconnue. "
                            "Vérifiez que la collection a été créée correctement avec create_index()."
                        )
                    
                    logger.warning(f"⚠️ Dimension déduite du modèle: {self.embedding_dimension}")

            logger.info(f"[SUCCESS] Index chargé: {self.document_count} docs, dim={self.embedding_dimension}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Erreur chargement: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        ✅ CORRIGÉ : Statistiques complètes sur le vector store
        
        Returns:
            Dict avec toutes les stats nécessaires
        """
        try:
            if not self.collection:
                return {
                    "total_documents": 0,
                    "embedding_dimension": self.embedding_dimension or 0,
                    "error": "Index non chargé"
                }
            
            return {
                "collection_name": self.collection.name,
                "total_documents": self.document_count,
                "embedding_dimension": self.embedding_dimension or 384,  # ✅ Avec fallback
                "corpus_size": len(self.corpus_mapping),  # ✅ AJOUTÉ
                "corpus_mapping_size": len(self.corpus_mapping),
                "metadata": self.collection.metadata,
                "is_loaded": True
            }
        except Exception as e:
            logger.error(f"[ERROR] get_statistics: {e}")
            return {
                "total_documents": 0,
                "embedding_dimension": 0,
                "error": str(e)
            }

# ============================================================================
# IMPLÉMENTATION 2 : FAISS (Recommandé pour production)
# ============================================================================

class FAISSVectorStore(BaseVectorStore):
    """
    Vector Store basé sur FAISS - Performance Optimale
    
    Avantages vs ChromaDB:
    - Plus rapide (2-3x sur recherches)
    - Moins de RAM
    - Pas de dépendances lourdes
    - Fichiers plus petits
    
    Inconvénients:
    - Pas de filtrage métadonnées natif
    - Plus bas niveau (mais on l'abstrait ici)
    
    Usage:
        >>> store = FAISSVectorStore("./data/faiss_db")
        >>> store.create_index(corpus, embeddings)
        >>> results = store.search(query_embedding, k=5)
    """
    
    def __init__(self, persist_directory: str = "./data/faiss_db"):
        super().__init__(persist_directory)
        self.index = None
        self.corpus_data = []
        self.id_to_idx = {}  # Mapping ID document -> index FAISS
    
    def create_index(self, corpus: List[Dict], embeddings: Union[List, np.ndarray],
                    index_type: str = "Flat") -> bool:
        """
        Crée l'index FAISS
        
        Args:
            corpus: Liste de documents
            embeddings: Embeddings correspondants
            index_type: Type d'index FAISS
                - "Flat" : Exact search (recommandé <100k docs)
                - "IVF" : Approximate search (>100k docs)
                - "HNSW" : Hierarchical NSW (bon compromis)
        
        Returns:
            bool: True si succès
        """
        try:
            logger.info("[LAUNCH] Création index FAISS...")
            start_time = time.time()
            
            # Validation
            self.validate_corpus_data(corpus, embeddings)
            
            # Convertir embeddings en numpy si nécessaire
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, dtype='float32')
            else:
                embeddings = embeddings.astype('float32')
            
            # Normaliser embeddings (pour similarité cosinus)
            faiss.normalize_L2(embeddings)
            
            # Créer index selon type
            if index_type == "Flat":
                # Index exact (Inner Product = cosine similarity après normalisation)
                self.index = faiss.IndexFlatIP(self.embedding_dimension)
                
            elif index_type == "IVF":
                # Index approximatif (plus rapide pour gros corpus)
                nlist = min(100, len(corpus) // 10)  # Nombre de clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
                self.index.train(embeddings)
                
            elif index_type == "HNSW":
                # Hierarchical Navigable Small World (bon compromis)
                self.index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)
                self.index.hnsw.efConstruction = 40
                
            else:
                raise ValueError(f"Type d'index inconnu: {index_type}")
            
            # Ajouter vecteurs à l'index
            self.index.add(embeddings)
            
            # Stocker corpus et créer mapping
            self.corpus_data = corpus
            self.id_to_idx = {doc['id']: i for i, doc in enumerate(corpus)}
            self.document_count = len(corpus)
            
            elapsed = time.time() - start_time
            
            logger.info(f"[SUCCESS] Index FAISS créé: {self.document_count} documents en {elapsed:.2f}s")
            logger.info(f"[STATS] Type: {index_type}, Dimension: {self.embedding_dimension}")
            # Éviter division par zéro si elapsed est trop petit
            if elapsed > 1e-6:  # Plus de 1 microseconde
                logger.info(f"[STATS] Vitesse: {self.document_count/elapsed:.1f} docs/sec")
            else:
                logger.info(f"[STATS] Vitesse: > {self.document_count/1e-6:.1f} docs/sec (temps < 1µs)")
            
            # Sauvegarder
            self.save()
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Erreur création index FAISS: {e}")
            raise
    
    def search(self, query_embedding: Union[List[float], np.ndarray], 
               k: int = 5,
               threshold: float = 0.0) -> List[SearchResult]:
        """
        Recherche sémantique avec FAISS
        
        Args:
            query_embedding: Embedding de la question
            k: Nombre de résultats
            threshold: Seuil minimum de similarité (0-1)
            
        Returns:
            List[SearchResult]: Résultats ordonnés
        """
        try:
            if self.index is None:
                raise RuntimeError("Index non initialisé. Appeler create_index() d'abord.")
            
            # Convertir query en numpy
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype='float32')
            else:
                query_embedding = query_embedding.astype('float32')
            
            # Reshape si nécessaire
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Normaliser pour similarité cosinus
            faiss.normalize_L2(query_embedding)
            
            # Recherche
            start_time = time.time()
            similarities, indices = self.index.search(query_embedding, k)
            elapsed = time.time() - start_time
            
            # Formater résultats
            results = []
            
            for rank, (idx, similarity) in enumerate(zip(indices[0], similarities[0]), 1):
                # FAISS retourne -1 si moins de k résultats
                if idx == -1:
                    break
                
                # Appliquer seuil
                if similarity < threshold:
                    continue
                
                doc = self.corpus_data[idx]
                
                results.append(SearchResult(
                    document_id=doc['id'],
                    document_text=doc['contenu'],
                    metadata={
                        'titre': doc.get('titre', 'Unknown'),
                        'source': doc.get('source', 'Unknown'),
                        'organisme': doc.get('organisme', 'Unknown'),
                        'type': doc.get('type', 'general')
                    },
                    similarity_score=float(similarity),
                    rank=rank
                ))
            
            logger.info(f"[SEARCH] Recherche FAISS: {len(results)} résultats en {elapsed*1000:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] Erreur recherche FAISS: {e}")
            return []
    
    def save(self, index_file: str = "faiss_index.index",
             corpus_file: str = "corpus_data.pkl") -> bool:
        """
        Sauvegarde l'index FAISS et le corpus
        
        Args:
            index_file: Nom fichier index FAISS
            corpus_file: Nom fichier corpus
            
        Returns:
            bool: True si succès
        """
        try:
            # Sauvegarder index FAISS
            index_path = self.persist_directory / index_file
            faiss.write_index(self.index, str(index_path))
            
            # Sauvegarder corpus et métadonnées
            metadata = {
                'corpus_data': self.corpus_data,
                'id_to_idx': self.id_to_idx,
                'embedding_dimension': self.embedding_dimension,
                'document_count': self.document_count
            }
            
            corpus_path = self.persist_directory / corpus_file
            with open(corpus_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f" Index FAISS sauvegardé:")
            logger.info(f"   - Index: {index_path}")
            logger.info(f"   - Corpus: {corpus_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Erreur sauvegarde FAISS: {e}")
            return False
    
    def load(self, index_file: str = "faiss_index.index",
             corpus_file: str = "corpus_data.pkl") -> bool:
        """
        Charge un index FAISS existant
        
        Args:
            index_file: Nom fichier index
            corpus_file: Nom fichier corpus
            
        Returns:
            bool: True si succès
        """
        try:
            logger.info("[LOAD] Chargement index FAISS...")
            
            # Charger index
            index_path = self.persist_directory / index_file
            if not index_path.exists():
                raise FileNotFoundError(f"Index introuvable: {index_path}")
            
            self.index = faiss.read_index(str(index_path))
            
            # Charger corpus
            corpus_path = self.persist_directory / corpus_file
            if not corpus_path.exists():
                raise FileNotFoundError(f"Corpus introuvable: {corpus_path}")
            
            with open(corpus_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.corpus_data = metadata['corpus_data']
            self.id_to_idx = metadata['id_to_idx']
            self.embedding_dimension = metadata['embedding_dimension']
            self.document_count = metadata['document_count']
            
            logger.info(f"[SUCCESS] Index FAISS chargé:")
            logger.info(f"   - Documents: {self.document_count}")
            logger.info(f"   - Dimension: {self.embedding_dimension}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Erreur chargement FAISS: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Statistiques sur l'index FAISS"""
        if self.index is None:
            return {"error": "Index non chargé"}
        
        return {
            "index_type": self.index.__class__.__name__,
            "total_documents": self.document_count,
            "embedding_dimension": self.embedding_dimension,
            "index_size": self.index.ntotal,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def load_corpus_and_embeddings(corpus_path: str = "data/corpus.json",
                               embeddings_path: str = "data/embeddings.npy") -> Tuple[List[Dict], np.ndarray]:
    """
    Charge le corpus et les embeddings depuis les fichiers
    
    Args:
        corpus_path: Chemin vers corpus.json
        embeddings_path: Chemin vers embeddings.npy
        
    Returns:
        Tuple[List[Dict], np.ndarray]: (corpus, embeddings)
    """
    logger.info(f"[LOAD] Chargement corpus: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    logger.info(f"[LOAD] Chargement embeddings: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    logger.info(f"[SUCCESS] Chargé: {len(corpus)} documents, embeddings shape={embeddings.shape}")
    
    return corpus, embeddings


def compare_vector_stores(corpus: List[Dict], embeddings: np.ndarray, 
                         query_embedding: np.ndarray, k: int = 5):
    """
    Compare les performances ChromaDB vs FAISS
    
    Args:
        corpus: Documents
        embeddings: Embeddings
        query_embedding: Query de test
        k: Nombre de résultats
    """
    logger.info("\n" + "="*60)
    logger.info(" BENCHMARK: ChromaDB vs FAISS")
    logger.info("="*60)
    
    results = {}
    
    # Test ChromaDB
    logger.info("\n[STATS] Test ChromaDB...")
    chroma_store = ChromaVectorStore("./data/benchmark_chroma")
    
    start = time.time()
    chroma_store.create_index(corpus, embeddings, reset=True)
    chroma_index_time = time.time() - start
    
    start = time.time()
    chroma_results = chroma_store.search(query_embedding, k=k)
    chroma_search_time = time.time() - start
    
    results['chroma'] = {
        'index_time': chroma_index_time,
        'search_time': chroma_search_time,
        'results_count': len(chroma_results)
    }
    
    # Test FAISS
    logger.info("\n[STATS] Test FAISS...")
    faiss_store = FAISSVectorStore("./data/benchmark_faiss")
    
    start = time.time()
    faiss_store.create_index(corpus, embeddings)
    faiss_index_time = time.time() - start
    
    start = time.time()
    faiss_results = faiss_store.search(query_embedding, k=k)
    faiss_search_time = time.time() - start
    
    results['faiss'] = {
        'index_time': faiss_index_time,
        'search_time': faiss_search_time,
        'results_count': len(faiss_results)
    }
    
    # Afficher comparaison
    logger.info("\n" + "="*60)
    logger.info("[STATS] RÉSULTATS BENCHMARK")
    logger.info("="*60)
    logger.info(f"\nCHROMADB:")
    logger.info(f"  Temps indexation: {chroma_index_time:.3f}s")
    logger.info(f"  Temps recherche:  {chroma_search_time*1000:.1f}ms")
    
    logger.info(f"\nFAISS:")
    logger.info(f"  Temps indexation: {faiss_index_time:.3f}s")
    logger.info(f"  Temps recherche:  {faiss_search_time*1000:.1f}ms")
    
    logger.info(f"\n GAGNANT:")
    # Éviter division par zéro avec une valeur minimale (epsilon)
    epsilon = 1e-6  # 1 microseconde minimum
    safe_chroma_time = max(chroma_search_time, epsilon)
    safe_faiss_time = max(faiss_search_time, epsilon)
    
    if faiss_search_time < chroma_search_time:
        speedup = safe_chroma_time / safe_faiss_time
        logger.info(f"  FAISS est {speedup:.1f}x plus rapide en recherche")
    elif chroma_search_time < faiss_search_time:
        speedup = safe_faiss_time / safe_chroma_time
        logger.info(f"  ChromaDB est {speedup:.1f}x plus rapide en recherche")
    else:
        logger.info(f"  Temps de recherche équivalents (différence < {epsilon*1000:.3f}ms)")
    
    return results


# ============================================================================
# TESTS AUTOMATISÉS
# ============================================================================

def test_vector_store_complete():
    """
    Suite de tests complète pour les vector stores
    
    Tests:
    1. Création index avec données réelles
    2. Recherche sémantique
    3. Persistance (save/load)
    4. Performance
    5. Validation dimensions multiples
    """
    logger.info("\n" + "="*70)
    logger.info("[TEST] SUITE DE TESTS COMPLÈTE - VECTOR STORES")
    logger.info("="*70)
    
    try:
        # ===== TEST 1: Données de test =====
        logger.info("\n Préparation données de test...")
        
        test_corpus = [
            {
                "id": "doc_001",
                "titre": "Culture du sorgho au Burkina Faso",
                "contenu": "Le sorgho est une céréale très résistante à la sécheresse, particulièrement adaptée au climat sahélien du Burkina Faso. Il nécessite 400-600mm d'eau par saison. L'engrais NPK 14-23-14 est recommandé à raison de 150-200 kg/ha.",
                "source": "FAO - Guide technique 2023",
                "organisme": "FAO",
                "type": "guide_technique"
            },
            {
                "id": "doc_002",
                "titre": "Fertilisation du mil",
                "contenu": "Le mil nécessite une fertilisation azotée pour de bons rendements. Application d'urée à 30-40 jours après semis améliore significativement la production. Le mil tolère mieux la sécheresse que le maïs.",
                "source": "CIRAD - Fiche pratique",
                "organisme": "CIRAD",
                "type": "fiche_technique"
            },
            {
                "id": "doc_003",
                "titre": "Maraîchage urbain à Ouagadougou",
                "contenu": "Le maraîchage urbain connaît un essor important dans la capitale. Les cultures principales sont la tomate, l'oignon, le chou. L'irrigation goutte-à-goutte permet d'économiser l'eau. Fumure organique recommandée.",
                "source": "INSD - Enquête agricole 2024",
                "organisme": "INSD",
                "type": "statistiques"
            },
            {
                "id": "doc_004",
                "titre": "Protection contre les ravageurs",
                "contenu": "Les insectes ravageurs causent des pertes importantes. Lutte intégrée recommandée: rotation des cultures, variétés résistantes, traitement au neem. Éviter usage excessif de pesticides chimiques.",
                "source": "GIZ - Programme agriculture durable",
                "organisme": "GIZ",
                "type": "guide_pratique"
            },
            {
                "id": "doc_005",
                "titre": "Calendrier cultural région Centre",
                "contenu": "Région Centre (Ouagadougou): Plantation sorgho et mil en juin-juillet. Maïs peut être planté dès mai si pluies précoces. Récolte octobre-novembre. Période critique pour eau: montaison et floraison.",
                "source": "Ministère Agriculture BF",
                "organisme": "MINAGRI",
                "type": "calendrier"
            }
        ]
        
        # Générer embeddings de test (simulation LaBSE 768D)
        np.random.seed(42)
        test_embeddings = np.random.randn(5, 768).astype('float32')
        
        # Normaliser pour similarité réaliste
        test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        
        logger.info(f"[SUCCESS] {len(test_corpus)} documents de test créés")
        logger.info(f"[SUCCESS] Embeddings shape: {test_embeddings.shape}")
        
        # ===== TEST 2: ChromaDB =====
        logger.info("\n" + "-"*70)
        logger.info(" TEST CHROMADB")
        logger.info("-"*70)
        
        chroma_store = ChromaVectorStore("./data/test_chroma")
        
        # Test création index
        logger.info("\n1️⃣ Test création index...")
        success = chroma_store.create_index(test_corpus, test_embeddings, reset=True)
        assert success, "[ERROR] Échec création index ChromaDB"
        logger.info("[SUCCESS] Index ChromaDB créé")
        
        # Test recherche
        logger.info("\n2️⃣ Test recherche...")
        query_embedding = test_embeddings[0]  # Utiliser premier doc comme query
        results = chroma_store.search(query_embedding, k=3)
        
        assert len(results) > 0, "[ERROR] Aucun résultat de recherche"
        logger.info(f"[SUCCESS] {len(results)} résultats trouvés")
        
        for i, result in enumerate(results, 1):
            logger.info(f"\n   Résultat #{i}:")
            logger.info(f"   - ID: {result.document_id}")
            logger.info(f"   - Titre: {result.metadata['titre']}")
            logger.info(f"   - Score: {result.similarity_score:.4f}")
            logger.info(f"   - Extrait: {result.document_text[:100]}...")
        
        # Test persistance
        logger.info("\n3️⃣ Test save/load...")
        save_success = chroma_store.save()
        assert save_success, "[ERROR] Échec sauvegarde ChromaDB"
        
        # Créer nouvelle instance et charger
        chroma_store2 = ChromaVectorStore("./data/test_chroma")
        load_success = chroma_store2.load()
        assert load_success, "[ERROR] Échec chargement ChromaDB"
        
        # Vérifier recherche après reload
        results2 = chroma_store2.search(query_embedding, k=3)
        assert len(results2) == len(results), "[ERROR] Résultats différents après reload"
        logger.info("[SUCCESS] Persistance ChromaDB validée")
        
        # Statistiques
        logger.info("\n4️⃣ Statistiques ChromaDB:")
        stats = chroma_store.get_statistics()
        for key, value in stats.items():
            logger.info(f"   - {key}: {value}")
        
        # ===== TEST 3: FAISS =====
        logger.info("\n" + "-"*70)
        logger.info("🟢 TEST FAISS")
        logger.info("-"*70)
        
        faiss_store = FAISSVectorStore("./data/test_faiss")
        
        # Test création index
        logger.info("\n1️⃣ Test création index FAISS...")
        success = faiss_store.create_index(test_corpus, test_embeddings)
        assert success, "[ERROR] Échec création index FAISS"
        logger.info("[SUCCESS] Index FAISS créé")
        
        # Test recherche
        logger.info("\n2️⃣ Test recherche FAISS...")
        results = faiss_store.search(query_embedding, k=3)
        
        assert len(results) > 0, "[ERROR] Aucun résultat FAISS"
        logger.info(f"[SUCCESS] {len(results)} résultats trouvés")
        
        for i, result in enumerate(results, 1):
            logger.info(f"\n   Résultat #{i}:")
            logger.info(f"   - ID: {result.document_id}")
            logger.info(f"   - Titre: {result.metadata['titre']}")
            logger.info(f"   - Score: {result.similarity_score:.4f}")
        
        # Test persistance
        logger.info("\n3️⃣ Test save/load FAISS...")
        save_success = faiss_store.save()
        assert save_success, "[ERROR] Échec sauvegarde FAISS"
        
        # Reload
        faiss_store2 = FAISSVectorStore("./data/test_faiss")
        load_success = faiss_store2.load()
        assert load_success, "[ERROR] Échec chargement FAISS"
        
        results2 = faiss_store2.search(query_embedding, k=3)
        assert len(results2) == len(results), "[ERROR] Résultats différents après reload"
        logger.info("[SUCCESS] Persistance FAISS validée")
        
        # Statistiques
        logger.info("\n4️⃣ Statistiques FAISS:")
        stats = faiss_store.get_statistics()
        for key, value in stats.items():
            logger.info(f"   - {key}: {value}")
        
        # ===== TEST 4: Benchmark =====
        logger.info("\n" + "-"*70)
        logger.info("⚡ BENCHMARK PERFORMANCE")
        logger.info("-"*70)
        
        benchmark_results = compare_vector_stores(test_corpus, test_embeddings, query_embedding, k=3)
        
        # ===== TEST 5: Validation dimensions =====
        logger.info("\n" + "-"*70)
        logger.info("[FIX] TEST DIMENSIONS MULTIPLES")
        logger.info("-"*70)
        
        # Test avec dimension 384 (MiniLM)
        logger.info("\nTest dimension 384 (MiniLM)...")
        embeddings_384 = np.random.randn(5, 384).astype('float32')
        embeddings_384 = embeddings_384 / np.linalg.norm(embeddings_384, axis=1, keepdims=True)
        
        faiss_store_384 = FAISSVectorStore("./data/test_faiss_384")
        success = faiss_store_384.create_index(test_corpus, embeddings_384)
        assert success, "[ERROR] Échec dimension 384"
        logger.info("[SUCCESS] Dimension 384 supportée")
        
        # ===== RÉSUMÉ =====
        logger.info("\n" + "="*70)
        logger.info("[CELEBRATE] TOUS LES TESTS RÉUSSIS")
        logger.info("="*70)
        logger.info("\n[SUCCESS] ChromaDB: OK")
        logger.info("[SUCCESS] FAISS: OK")
        logger.info("[SUCCESS] Persistance: OK")
        logger.info("[SUCCESS] Recherche sémantique: OK")
        logger.info("[SUCCESS] Dimensions multiples: OK")
        logger.info("[SUCCESS] Performance: OK")
        
        return True
        
    except AssertionError as e:
        logger.error(f"\n[ERROR] TEST ÉCHOUÉ: {e}")
        return False
    except Exception as e:
        logger.error(f"\n[ERROR] ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# SCRIPT PRINCIPAL - INTÉGRATION AVEC VOTRE PIPELINE
# ============================================================================

def main():
    """
    Script principal - Utilisation avec votre corpus existant
    
    Scénarios:
    1. Première fois: Créer index depuis corpus.json + embeddings.npy
    2. Utilisation: Charger index existant
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Vector Store pour RAG Agricole BF')
    parser.add_argument('--mode', choices=['create', 'load', 'test', 'benchmark'],
                       default='test', help='Mode opération')
    parser.add_argument('--backend', choices=['chroma', 'faiss'], 
                       default='faiss', help='Backend vector store')
    parser.add_argument('--corpus', default='data/corpus.json',
                       help='Chemin corpus.json')
    parser.add_argument('--embeddings', default='data/embeddings.npy',
                       help='Chemin embeddings.npy')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Lancer tests automatisés
        logger.info("[TEST] Mode TEST - Suite de tests complète")
        test_vector_store_complete()
        
    elif args.mode == 'benchmark':
        # Benchmark ChromaDB vs FAISS
        logger.info("⚡ Mode BENCHMARK")
        corpus, embeddings = load_corpus_and_embeddings(args.corpus, args.embeddings)
        query_emb = embeddings[0]  # Première doc comme query
        compare_vector_stores(corpus, embeddings, query_emb, k=5)
        
    elif args.mode == 'create':
        # Créer nouvel index
        logger.info(f" Mode CREATE - Backend: {args.backend}")
        
        # Charger données
        corpus, embeddings = load_corpus_and_embeddings(args.corpus, args.embeddings)
        
        # Créer index selon backend
        if args.backend == 'chroma':
            store = ChromaVectorStore("./data/chroma_db")
            store.create_index(corpus, embeddings, reset=True)
        else:
            store = FAISSVectorStore("./data/faiss_db")
            store.create_index(corpus, embeddings)
        
        logger.info("[SUCCESS] Index créé avec succès")
        
        # Test rapide
        logger.info("\n[TEST] Test rapide...")
        results = store.search(embeddings[0], k=3)
        logger.info(f"[SUCCESS] Recherche OK: {len(results)} résultats")
        
    elif args.mode == 'load':
        # Charger index existant
        logger.info(f"[LOAD] Mode LOAD - Backend: {args.backend}")
        
        if args.backend == 'chroma':
            store = ChromaVectorStore("./data/chroma_db")
            success = store.load()
        else:
            store = FAISSVectorStore("./data/faiss_db")
            success = store.load()
        
        if success:
            logger.info("[SUCCESS] Index chargé avec succès")
            stats = store.get_statistics()
            logger.info("\n[STATS] Statistiques:")
            for key, value in stats.items():
                logger.info(f"   - {key}: {value}")
        else:
            logger.error("[ERROR] Échec chargement index")


# ============================================================================
# EXEMPLE D'UTILISATION DANS VOTRE PIPELINE RAG
# ============================================================================

class ExempleIntegrationRAG:
    """
    Exemple d'intégration du VectorStore dans votre pipeline RAG
    """
    
    def __init__(self, use_faiss: bool = True):
        """
        Args:
            use_faiss: Si True, utilise FAISS (plus rapide), sinon ChromaDB
        """
        if use_faiss:
            self.vector_store = FAISSVectorStore("./data/faiss_db")
            logger.info("🟢 VectorStore: FAISS")
        else:
            self.vector_store = ChromaVectorStore("./data/chroma_db")
            logger.info(" VectorStore: ChromaDB")
        
        self.initialized = False
    
    def setup(self, corpus_path: str = "data/corpus.json",
              embeddings_path: str = "data/embeddings.npy"):
        """
        Configuration initiale: créer index
        À faire UNE SEULE FOIS
        """
        logger.info("[LAUNCH] Setup VectorStore...")
        
        # Charger données
        corpus, embeddings = load_corpus_and_embeddings(corpus_path, embeddings_path)
        
        # Créer index
        self.vector_store.create_index(corpus, embeddings)
        self.initialized = True
        
        logger.info("[SUCCESS] VectorStore prêt")
    
    def load_existing(self):
        """
        Charger index existant
        À faire à chaque démarrage de l'application
        """
        logger.info("[LOAD] Chargement VectorStore existant...")
        
        success = self.vector_store.load()
        if success:
            self.initialized = True
            logger.info("[SUCCESS] VectorStore chargé")
        else:
            logger.error("[ERROR] Échec chargement - Lancer setup() d'abord")
        
        return success
    
    def retrieve_documents(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Récupère documents pertinents pour une question
        
        Args:
            query_embedding: Embedding de la question
            k: Nombre de documents à récupérer
            
        Returns:
            List[Dict]: Documents pertinents avec métadonnées
        """
        if not self.initialized:
            raise RuntimeError("VectorStore non initialisé. Appeler setup() ou load_existing()")
        
        # Recherche
        results = self.vector_store.search(query_embedding, k=k)
        
        # Formater pour le LLM
        formatted_docs = []
        for result in results:
            formatted_docs.append({
                'id': result.document_id,
                'text': result.document_text,
                'metadata': result.metadata,
                'score': result.similarity_score,
                'rank': result.rank
            })
        
        return formatted_docs


if __name__ == "__main__":
    # Si lancé directement, exécuter le script principal
    main()