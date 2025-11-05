"""
FastAPI Server - Production Ready
API REST pour Système RAG Agricole Burkina Faso

Auteur: Expert ML Team  
Date: 3 Novembre 2025
Hackathon: MTDPCE 2025

Fonctionnalités:
- Endpoints /ask, /health, /system/info
- Validation Pydantic
- CORS configuré
- Rate limiting
- Cache intelligent
- Tests automatisés
- Intégration avec modules optimisés (FAISS + nouveau LLM)

Usage:
    # Développement
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    
    # Production
    uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
"""

import logging
import sys
import io
import os
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from functools import lru_cache

# Forcer la sortie console en UTF-8 (évite les erreurs Unicode sur Windows)
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

# Ajouter src au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

# Import modules optimisés
from src.embeddings import EmbeddingPipeline
from src.vector_store import FAISSVectorStore  # Ou ChromaVectorStore
from src.llm_handler import LLMHandler, PromptTemplate, GenerationConfig, LLMBackend

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS (Validation requêtes/réponses)
# ============================================================================

class AskRequest(BaseModel):
    """
    Modèle de requête pour poser une question
    
    Validation automatique:
    - question entre 5 et 500 caractères
    - max_results entre 1 et 10
    - template parmi les valeurs valides
    """
    question: str = Field(
        ..., 
        min_length=5, 
        max_length=500,
        description="Question agricole en français",
        example="Quel engrais utiliser pour le mil en saison sèche ?"
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre de documents de contexte à utiliser"
    )
    template: str = Field(
        default="standard",
        description="Template de prompt (standard, concise, detailed)",
        example="standard"
    )
    verbose: bool = Field(
        default=False,
        description="Afficher détails de traitement"
    )
    
    @field_validator('template')
    @classmethod
    def validate_template(cls, v):
        """Valide que le template est supporté"""
        valid = ['standard', 'concise', 'detailed']
        if v.lower() not in valid:
            raise ValueError(f"Template doit être parmi {valid}")
        return v.lower()
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Quel engrais utiliser pour le mil en saison sèche ?",
                "max_results": 3,
                "template": "standard",
                "verbose": False
            }
        }
    )


class SourceInfo(BaseModel):
    """Information sur une source documentaire"""
    titre: str
    source: str
    organisme: str
    pertinence: float = Field(ge=0, le=1)


class AskResponse(BaseModel):
    """
    Modèle de réponse pour une question
    
    Contient:
    - La réponse générée
    - Les sources utilisées
    - Les métadonnées de performance
    """
    success: bool
    question: str
    reponse: str
    sources: List[SourceInfo]
    metadata: Dict[str, Any]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "question": "Quel engrais pour le mil ?",
                "reponse": "Pour la culture du mil au Burkina Faso, l'engrais NPK 14-23-14 est recommandé...",
                "sources": [
                    {
                        "titre": "Guide culture mil",
                        "source": "FAO 2023",
                        "organisme": "FAO",
                        "pertinence": 0.89
                    }
                ],
                "metadata": {
                    "backend": "ollama",
                    "model": "mistral:7b",
                    "generation_time": 2.3,
                    "tokens_generated": 156,
                    "documents_used": 3,
                    "processing_time": 2.5
                }
            }
        }
    )


class HealthResponse(BaseModel):
    """Réponse health check"""
    status: str
    timestamp: str
    version: str
    components: Dict[str, Any]
    uptime: float


class SystemInfoResponse(BaseModel):
    """Informations système"""
    system_name: str
    version: str
    description: str
    uptime: float
    components: Dict[str, Any]
    endpoints: List[Dict[str, str]]
    statistics: Dict[str, Any]


# ============================================================================
# CLASSE API PRINCIPALE
# ============================================================================

class AgriculturalRAGAPI:
    """
    API REST Production-Ready pour RAG Agricole BF
    
    Architecture:
    - FastAPI avec validation Pydantic
    - Rate limiting (10 req/min)
    - Cache intelligent
    - Logging complet
    - Tests automatisés
    
    Intégration:
    - EmbeddingGenerator (votre module)
    - FAISSVectorStore (votre module optimisé)
    - LLMHandler (votre module optimisé)
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        use_faiss: bool = True,
        enable_cache: bool = True,
        enable_rate_limit: bool = True
    ):
        """
        Initialise l'API
        
        Args:
            use_faiss: Utiliser FAISS (True) ou ChromaDB (False)
            enable_cache: Activer cache réponses
            enable_rate_limit: Activer limitation débit
        """
        # Configuration
        self.use_faiss = use_faiss
        self.enable_cache = enable_cache
        self.enable_rate_limit = enable_rate_limit
        
        # FastAPI app
        self.app = FastAPI(
            title="API RAG Agricole Burkina Faso",
            description=(
                "API REST professionnelle pour questions-réponses sur l'agriculture burkinabè. "
                "Utilise RAG (Retrieval-Augmented Generation) avec données locales."
            ),
            version=self.VERSION,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Rate limiter
        if self.enable_rate_limit:
            self.limiter = Limiter(key_func=get_remote_address)
            self.app.state.limiter = self.limiter
            self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Composants système (initialisés au startup)
        self.embedding_model = None
        self.vector_store = None
        self.llm_handler = None
        
        # Métadonnées
        self.startup_time = datetime.now()
        self.request_count = 0
        self.cache_hits = 0
        
        # Setup
        self._setup_middleware()
        self._setup_routes()
        self._setup_events()
        
        # Cache des réponses
        self._response_cache: Dict[str, Tuple[AskResponse, float]] = {}
        self.cache_ttl = 3600  # 1 heure
    
    def _setup_middleware(self):
        """Configure middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Middleware de logging personnalisé
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log toutes les requêtes"""
            start_time = time.time()
            
            response = await call_next(request)
            
            process_time = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            return response
    
    def _normalize_corpus(self, corpus: List[Dict], text_field: str = None) -> List[Dict]:
        """
        Normalise le corpus pour correspondre au format attendu par vector_store.
        
        Transforme:
        - chunk_id -> id
        - text/contenu -> contenu
        - Génère titre si manquant
        - Assure présence de source
        """
        if not corpus:
            return []
        
        # Détecter champ texte si non fourni (une seule fois)
        if text_field is None:
            for field in ['text', 'contenu', 'texte', 'content']:
                if field in corpus[0] and corpus[0].get(field):
                    text_field = field
                    break
        
        # Si toujours pas trouvé, utiliser le premier champ texte long
        if text_field is None:
            for key, value in corpus[0].items():
                if isinstance(value, str) and len(value) > 50:
                    text_field = key
                    break
        
        normalized = []
        
        for i, doc in enumerate(corpus):
            # Extraire le contenu texte (avec fallback)
            contenu = ''
            if text_field and text_field in doc:
                contenu = doc[text_field] or ''
            if not contenu:
                contenu = doc.get('contenu') or doc.get('text') or doc.get('texte') or doc.get('content') or ''
            
            # Créer document normalisé
            normalized_doc = {
                'id': doc.get('id') or doc.get('chunk_id') or f"doc_{i}",
                'titre': doc.get('titre') or doc.get('title') or doc.get('source', 'Document')[:100],
                'contenu': contenu,
                'source': doc.get('source') or doc.get('source_institution') or 'Unknown',
                'organisme': doc.get('organisme') or doc.get('source_institution') or 'Unknown',
                'type': doc.get('type') or 'general'
            }
            
            # Ajouter métadonnées supplémentaires si présentes
            if 'chunk_index' in doc:
                normalized_doc['chunk_index'] = doc['chunk_index']
            if 'total_chunks' in doc:
                normalized_doc['total_chunks'] = doc['total_chunks']
            if 'source_url' in doc:
                normalized_doc['source_url'] = doc['source_url']
            
            normalized.append(normalized_doc)
        
        return normalized
    
    def _setup_events(self):
        """Configure événements startup/shutdown"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialise composants au démarrage"""

            import os
            from dotenv import load_dotenv

            load_dotenv()

            # Au début de startup_event()
            auto_create_index = os.environ.get('AUTO_CREATE_INDEX', 'true').lower() == 'true'
            environment = os.environ.get('ENVIRONMENT', 'development')

            logger.info(f"[ENV] Environnement: {environment}")
            logger.info(f"[ENV] Auto-creation index: {auto_create_index}")
            logger.info("="*70)
            logger.info("[LAUNCH] DEMARRAGE API RAG AGRICOLE BURKINA FASO")
            logger.info("="*70)
            
            try:
                # 1. Embedding model
                logger.info("[STATS] Chargement modele embeddings...")
                self.embedding_model = EmbeddingPipeline("./data/corpus.json")
                self.embedding_model.initialize_embedding_model()
                logger.info(f"[SUCCESS] Embeddings: {self.embedding_model.model_name}")
                
                # 2. Vector store
                logger.info(f"[DOCS] Chargement vector store ({'FAISS' if self.use_faiss else 'ChromaDB'})...")
                
                if self.use_faiss:
                    from src.vector_store import FAISSVectorStore
                    self.vector_store = FAISSVectorStore("./data/faiss_db")
                else:
                    from src.vector_store import ChromaVectorStore
                    self.vector_store = ChromaVectorStore("./data/chroma_db")
                
                # ✅ AMÉLIORATION : Vérification intelligente
                from pathlib import Path
                import os
                
                # Déterminer si on est en mode développement ou production
                is_dev_mode = os.environ.get('ENVIRONMENT', 'development') == 'development'
                
                if self.use_faiss:
                    index_exists = (Path("./data/faiss_db/faiss_index.index").exists() and 
                                Path("./data/faiss_db/corpus_data.pkl").exists())
                else:
                    index_exists = Path("./data/chroma_db/corpus_mapping.pkl").exists()
                
                embeddings_path = "./data/embeddings.npy"
                embeddings_exist = Path(embeddings_path).exists()
                
                if index_exists:
                    # ✅ CAS 1 : Index existe déjà (mode normal)
                    logger.info("[LOAD] Index existant detecte, chargement...")
                    success = self.vector_store.load()
                    if success:
                        stats = self.vector_store.get_statistics()
                        logger.info(f"[SUCCESS] Vector store charge: {stats.get('total_documents', 0)} documents")
                    else:
                        logger.error("[ERROR] Echec chargement vector store")
                        
                elif embeddings_exist:
                    # ✅ CAS 2 : Embeddings existent mais pas l'index (recréation rapide)
                    logger.info("[REBUILD] Embeddings trouves, reconstruction de l'index...")
                    
                    import json
                    import numpy as np
                    
                    with open("./data/corpus.json", 'r', encoding='utf-8') as f:
                        corpus_raw = json.load(f)
                    
                    # Normaliser le corpus pour correspondre au format attendu
                    logger.info("[NORMALIZE] Normalisation du corpus...")
                    corpus = self._normalize_corpus(corpus_raw)
                    logger.info(f"[SUCCESS] Corpus normalise: {len(corpus)} documents")
                    
                    embeddings = np.load(embeddings_path)
                    logger.info(f"[SUCCESS] Embeddings charges: {embeddings.shape}")
                    
                    # Créer l'index
                    if self.use_faiss:
                        self.vector_store.create_index(corpus, embeddings)
                    else:
                        self.vector_store.create_index(corpus, embeddings, reset=True)
                    
                    logger.info(f"[SUCCESS] Index reconstruit: {len(corpus)} documents")
                    
                else:
                    # ✅ CAS 3 : Premier démarrage - Création complète
                    if is_dev_mode:
                        logger.warning("[SETUP] Premier demarrage detecte - Creation de l'index...")
                    else:
                        logger.info("[SETUP] Initialisation production - Generation de l'index...")
                    
                    import json
                    import numpy as np
                    
                    corpus_path = "./data/corpus.json"
                    
                    # Charger corpus
                    logger.info(f"[LOAD] Chargement corpus: {corpus_path}")
                    with open(corpus_path, 'r', encoding='utf-8') as f:
                        corpus = json.load(f)
                    logger.info(f"[SUCCESS] Corpus charge: {len(corpus)} documents")
                    
                    # Détecter champ texte
                    if corpus:
                        logger.info("[DEBUG] Structure du premier document:")
                        logger.info(f"[DEBUG] Cles disponibles: {list(corpus[0].keys())}")
                    
                    text_field = None
                    possible_fields = ['contenu', 'content', 'text', 'texte', 'document', 'body', 'description']
                    
                    for field in possible_fields:
                        if field in corpus[0]:
                            text_field = field
                            logger.info(f"[SUCCESS] Champ texte detecte: '{text_field}'")
                            break
                    
                    if not text_field:
                        for key, value in corpus[0].items():
                            if isinstance(value, str) and len(value) > 50:
                                text_field = key
                                logger.info(f"[SUCCESS] Champ texte auto-detecte: '{text_field}'")
                                break
                    
                    if not text_field:
                        raise ValueError(f"Impossible de trouver un champ texte. Cles: {list(corpus[0].keys())}")
                    
                    # Générer embeddings
                    logger.info("[GENERATE] Generation des embeddings pour le corpus...")
                    logger.info("[INFO] Cela peut prendre quelques minutes (une seule fois)...")
                    
                    texts = [doc.get(text_field, '') for doc in corpus if doc.get(text_field)]
                    
                    if not texts:
                        raise ValueError(f"Aucun texte trouve dans le champ '{text_field}'")
                    
                    embeddings = self.embedding_model.embedding_model.encode(
                        texts,
                        batch_size=32,
                        show_progress_bar=True
                    )
                    
                    # Sauvegarder embeddings
                    np.save(embeddings_path, embeddings)
                    logger.info(f"[SUCCESS] Embeddings sauvegardes: {embeddings.shape}")
                    
                    # Normaliser le corpus pour correspondre au format attendu
                    logger.info("[NORMALIZE] Normalisation du corpus...")
                    corpus_normalized = self._normalize_corpus(corpus, text_field=text_field)
                    logger.info(f"[SUCCESS] Corpus normalise: {len(corpus_normalized)} documents")
                    
                    # Créer l'index
                    logger.info("[CREATE] Creation de l'index vectoriel...")
                    if self.use_faiss:
                        self.vector_store.create_index(corpus_normalized, embeddings)
                    else:
                        self.vector_store.create_index(corpus_normalized, embeddings, reset=True)
                    
                    logger.info(f"[SUCCESS] Index cree: {len(corpus_normalized)} documents")
                    logger.info("[INFO] Les prochains demarrages seront beaucoup plus rapides!")
                
                # 3. LLM Handler
                logger.info("[LLM] Initialisation LLM Handler...")
                self.llm_handler = LLMHandler(
                    backend=LLMBackend.OLLAMA,
                    ollama_model="llama3.2:3b",
                    generation_config=GenerationConfig(
                        temperature=0.1,
                        max_tokens=800,
                        repeat_penalty=1.2,
                        top_p=0.9,
                        top_k=40
                    ),
                    enable_cache=self.enable_cache
                )
                
                health = self.llm_handler.health_check()
                logger.info(f"[SUCCESS] LLM Backend: {health['active_backend']}")
                
                logger.info("="*70)
                logger.info("[SUCCESS] API PRETE - Tous les composants initialises")
                logger.info(f"[API] Swagger UI: http://0.0.0.0:8000/docs")
                logger.info("="*70)
                
            except Exception as e:
                logger.error(f"[ERROR] ECHEC INITIALISATION: {e}")
                import traceback
                traceback.print_exc()
                logger.error("[WARNING] L'API demarrera en mode degrade")


        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup au shutdown"""
            logger.info("[STOP] Arret API - Nettoyage ressources...")
            
            # Statistiques finales
            uptime = (datetime.now() - self.startup_time).total_seconds()
            logger.info(f"[STATS] Statistiques finales:")
            logger.info(f"   Uptime: {uptime:.1f}s")
            logger.info(f"   Requetes totales: {self.request_count}")
            if self.enable_cache:
                cache_rate = (self.cache_hits / self.request_count * 100) if self.request_count > 0 else 0
                logger.info(f"   Cache hits: {self.cache_hits} ({cache_rate:.1f}%)")
    
    def _setup_routes(self):
        """Définit les routes API"""
        
        # Page d'accueil
        @self.app.get("/", tags=["System"])
        async def root():
            """Page d'accueil de l'API"""
            return {
                "message": "API RAG Agricole Burkina Faso",
                "version": self.VERSION,
                "status": "operational",
                "documentation": "/docs",
                "health_check": "/health",
                "description": "API pour questions-reponses sur l'agriculture burkinabe"
            }
        
        # Endpoint principal ASK
        if self.enable_rate_limit:
            @self.app.post("/ask", response_model=AskResponse, tags=["Q&A"])
            @self.limiter.limit("10/minute")
            async def ask_question(request: Request, ask_request: AskRequest):
                return await self._handle_ask(ask_request)
        else:
            @self.app.post("/ask", response_model=AskResponse, tags=["Q&A"])
            async def ask_question(ask_request: AskRequest):
                return await self._handle_ask(ask_request)
        
        # Health check
        @self.app.get("/health", response_model=HealthResponse, tags=["System"])
        async def health_check():
            """Vérifie santé de l'API et de ses composants"""
            try:
                components_status = {}
                
                # Embedding model
                if self.embedding_model and self.embedding_model.embedding_model:
                    try:
                        dimension = self.embedding_model.embedding_model.get_sentence_embedding_dimension()
                        components_status['embedding_model'] = {
                            'status': 'healthy',
                            'model': self.embedding_model.model_name,
                            'dimension': dimension
                        }
                    except Exception as e:
                        components_status['embedding_model'] = {
                            'status': 'error',
                            'error': str(e)
                        }
                else:
                    components_status['embedding_model'] = {'status': 'not_initialized'}
                
                # Vector store
                if self.vector_store:
                    stats = self.vector_store.get_statistics()
                    components_status['vector_store'] = {
                        'status': 'healthy',
                        'type': 'FAISS' if self.use_faiss else 'ChromaDB',
                        'documents': stats.get('total_documents', 0),
                        'dimension': stats.get('embedding_dimension', 0)
                    }
                else:
                    components_status['vector_store'] = {'status': 'not_initialized'}
                
                # LLM Handler
                if self.llm_handler:
                    llm_health = self.llm_handler.health_check()
                    components_status['llm_handler'] = {
                        'status': 'healthy' if llm_health['active_backend'] else 'degraded',
                        'backend': llm_health['active_backend'],
                        'ollama': llm_health['ollama'].get('status', 'unknown'),
                        'huggingface': llm_health['huggingface'].get('status', 'unknown')
                    }
                else:
                    components_status['llm_handler'] = {'status': 'not_initialized'}
                
                # Statut global
                all_healthy = all(
                    comp.get('status') == 'healthy' 
                    for comp in components_status.values()
                )
                overall_status = 'healthy' if all_healthy else 'degraded'
                
                uptime = (datetime.now() - self.startup_time).total_seconds()
                
                return {
                    'status': overall_status,
                    'timestamp': datetime.now().isoformat(),
                    'version': self.VERSION,
                    'components': components_status,
                    'uptime': uptime
                }
                
            except Exception as e:
                logger.error(f"Erreur health check: {e}")
                return {
                    'status': 'unhealthy',
                    'timestamp': datetime.now().isoformat(),
                    'version': self.VERSION,
                    'components': {'error': str(e)},
                    'uptime': 0
                }
        
        # System info
        @self.app.get("/system/info", response_model=SystemInfoResponse, tags=["System"])
        async def system_info():
            """Informations détaillées sur le système"""
            try:
                uptime = (datetime.now() - self.startup_time).total_seconds()
                
                llm_stats = {}
                if self.llm_handler:
                    llm_stats = self.llm_handler.get_statistics()
                
                return {
                    'system_name': "API RAG Agricole Burkina Faso",
                    'version': self.VERSION,
                    'description': "Systeme de questions-reponses base sur RAG pour l'agriculture burkinabe",
                    'uptime': uptime,
                    'components': {
                        'embedding_model': self.embedding_model.model_name if self.embedding_model else None,
                        'vector_store': 'FAISS' if self.use_faiss else 'ChromaDB',
                        'llm_backend': self.llm_handler.active_backend.value if self.llm_handler else None,
                        'cache_enabled': self.enable_cache,
                        'rate_limit_enabled': self.enable_rate_limit
                    },
                    'endpoints': [
                        {'method': 'POST', 'path': '/ask', 'description': 'Poser une question agricole'},
                        {'method': 'GET', 'path': '/health', 'description': 'Verifier sante systeme'},
                        {'method': 'GET', 'path': '/system/info', 'description': 'Informations systeme'},
                        {'method': 'GET', 'path': '/docs', 'description': 'Documentation Swagger'},
                    ],
                    'statistics': {
                        'total_requests': self.request_count,
                        'cache_hits': self.cache_hits,
                        'cache_hit_rate': (self.cache_hits / self.request_count * 100) if self.request_count > 0 else 0,
                        **llm_stats
                    }
                }
                
            except Exception as e:
                logger.error(f"Erreur system info: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
        
        # Handlers d'erreurs
        @self.app.exception_handler(500)
        async def internal_error_handler(request, exc):
            logger.error(f"Erreur 500: {exc}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Erreur interne du serveur", "success": False}
            )
        
        @self.app.exception_handler(404)
        async def not_found_handler(request, exc):
            return JSONResponse(
                status_code=404,
                content={"detail": "Endpoint non trouve", "success": False}
            )

    async def _handle_ask(self, ask_request: AskRequest) -> AskResponse:
        """Traite une question utilisateur (logique métier)"""
        try:
            self.request_count += 1
            start_time = time.time()
            
            logger.info(f"[Q] Question recue: {ask_request.question}")
            
            # Vérifier système prêt
            if not self._is_system_ready():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Systeme non initialise. Verifier /health"
                )
            
            # Check cache
            if self.enable_cache:
                cache_key = self._get_cache_key(ask_request.question, ask_request.max_results)
                cached_response = self._get_from_cache(cache_key)
                if cached_response:
                    self.cache_hits += 1
                    logger.info("[CACHE] Reponse depuis cache")
                    return cached_response
            
            # 1. Embeddings de la question
            if ask_request.verbose:
                logger.info("[SEARCH] Generation embeddings question...")
            
            question_embedding = self.embedding_model.embedding_model.encode([ask_request.question])
            if len(question_embedding.shape) > 1:
                question_embedding = question_embedding[0]
            
            # 2. Recherche vectorielle
            if ask_request.verbose:
                logger.info(f"[DOCS] Recherche {ask_request.max_results} documents pertinents...")
            
            search_results = self.vector_store.search(
                query_embedding=question_embedding,
                k=ask_request.max_results
            )
            
            if not search_results:
                logger.warning("[WARNING] Aucun document pertinent trouve")
            
            # 3. Préparer contexte pour LLM
            context_docs = []
            sources_info = []
            
            for result in search_results:
                context_docs.append({
                    'text': result.document_text,
                    'metadata': result.metadata
                })
                
                sources_info.append(SourceInfo(
                    titre=result.metadata.get('titre', 'Document inconnu'),
                    source=result.metadata.get('source', 'Source inconnue'),
                    organisme=result.metadata.get('organisme', 'N/A'),
                    pertinence=float(result.similarity_score)
                ))
                
                if ask_request.verbose:
                    logger.info(f"   - {result.metadata.get('titre', 'Doc')} (score: {result.similarity_score:.3f})")
            
            # 4. Template mapping
            template_map = {
                'standard': PromptTemplate.STANDARD,
                'concise': PromptTemplate.CONCISE,
                'detailed': PromptTemplate.DETAILED
            }
            template = template_map.get(ask_request.template, PromptTemplate.STANDARD)
            
            # 5. Génération LLM
            if ask_request.verbose:
                logger.info("[LLM] Generation reponse...")
            
            llm_response = self.llm_handler.generate_answer(
                ask_request.question,
                context_docs,
                template=template,
                #use_simple_prompt=True
            )
            
            processing_time = time.time() - start_time
            
            # 6. Construire réponse
            response = AskResponse(
                success=llm_response.success,
                question=ask_request.question,
                reponse=llm_response.text,
                sources=sources_info,
                metadata={
                    'backend': llm_response.backend,
                    'model': llm_response.model,
                    'generation_time': llm_response.generation_time,
                    'tokens_generated': llm_response.tokens_generated,
                    'tokens_per_second': llm_response.tokens_per_second,
                    'documents_used': len(search_results),
                    'processing_time': processing_time,
                    'template': ask_request.template,
                    'cached': False
                }
            )
            
            # Cache
            if self.enable_cache:
                self._save_to_cache(cache_key, response)
            
            logger.info(f"[SUCCESS] Reponse generee en {processing_time:.2f}s")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[ERROR] Erreur traitement: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur: {str(e)}"
            )
    
    def _is_system_ready(self) -> bool:
        """Vérifie que tous les composants sont prêts"""
        return all([
            self.embedding_model is not None,
            self.vector_store is not None,
            self.llm_handler is not None
        ])
    
    def _get_cache_key(self, question: str, max_results: int) -> str:
        """Génère clé de cache"""
        cache_str = f"{question}|{max_results}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[AskResponse]:
        """Récupère depuis cache avec TTL"""
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return response
            else:
                del self._response_cache[cache_key]
        return None

    def _save_to_cache(self, cache_key: str, response: AskResponse):
        """Sauvegarde en cache avec timestamp"""
        self._response_cache[cache_key] = (response, time.time())
        if len(self._response_cache) > 100:
            oldest_key = min(self._response_cache.keys(), 
                            key=lambda k: self._response_cache[k][1])
            del self._response_cache[oldest_key]

# ============================================================================
# CRÉATION INSTANCE API
# ============================================================================

# Instance globale
api_instance = AgriculturalRAGAPI(
    use_faiss=True,            # Utiliser FAISS (plus rapide)
    enable_cache=True,          # Activer cache
    enable_rate_limit=True      # Activer rate limiting
)

app = api_instance.app


# ============================================================================
# FONCTION DE DÉMARRAGE
# ============================================================================

def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1
):
    """
    Démarre le serveur FastAPI
    
    Args:
        host: Adresse hôte
        port: Port
        reload: Auto-reload en développement
        workers: Nombre de workers (production)
    """
    logger.info(f"[LAUNCH] Démarrage serveur sur {host}:{port}")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Workers incompatibles avec reload
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    # Démarrage direct
    start_server(
        host="0.0.0.0",
        port=8000,
        reload=True  # Mode développement
    )