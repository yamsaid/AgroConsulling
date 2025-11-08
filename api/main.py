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
- Détection d'intention robuste
- Tests automatisés
"""

import logging
import sys
import io
import os
import time
import csv
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from difflib import SequenceMatcher

# Forcer la sortie console en UTF-8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    elif hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
except Exception:
    pass

# Ajouter src au path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

from src.embeddings import EmbeddingPipeline
from src.vector_store import FAISSVectorStore
from src.llm_handler import LLMHandler, PromptTemplate, GenerationConfig, LLMBackend

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api_server.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# DÉTECTION D'INTENTION AVEC FUZZY MATCHING
# ============================================================================


class IntentType(Enum):
    """Types d'intentions détectées"""

    GREETING = "greeting"
    THANKS = "thanks"
    SELF_DESCRIPTION = "self_description"
    OUT_OF_DOMAIN = "out_of_domain"
    AGRICULTURE = "agriculture"


class IntentDetector:
    """Détecte l'intention de la question avec fuzzy matching"""

    GREETING_KEYWORDS = [
        "bonjour",
        "hello",
        "salut",
        "hi",
        "hey",
        "coucou",
        "allo",
        "bonsoir",
        "good evening",
        "bonne nuit",
        "salutations",
        "greetings",
        "howdy",
        "comment ça va",
        "ça va",
        "how are you",
        "sup",
        "yo",
        "wesh",
        "slt",
        "cc",
        "matin",
        "bon matin",
        "alamousso",
    ]

    THANKS_KEYWORDS = [
        "merci",
        "thank you",
        "thanks",
        "gracias",
        "ta",
        "tq",
        "grâce",
        "gratitude",
        "appreci",
        "gentil",
        "aimable",
        "kind",
        "reconnaissance",
        "remercier",
        "merci bien",
        "merci beaucoup",
        "thank you very much",
        "sympa",
        "cool",
    ]

    SELF_DESCRIPTION_KEYWORDS = [
        "qui es-tu",
        "who are you",
        "c'est quoi",
        "what are you",
        "dis-moi qui tu es",
        "tell me about you",
        "parle de toi",
        "présente-toi",
        "introduce yourself",
        "quel est ton nom",
        "qu'est-ce que tu fais",
        "what do you do",
        "ton rôle",
        "role",
        "à quoi tu sers",
        "what is your purpose",
        "ton objectif",
        "qu'est-ce que tu es",
        "tes capacités",
        "capabilities",
        "comment tu marches",
        "how do you work",
        "explique-toi",
        "parle de ton système",
        "about your system",
        "décris-toi",
        "ton nom",
        "ta fonction",
        "ce que tu fais",
        "what can you do",
        "features",
        "fonctionnalités",
        "caracteristiques",
        "qui tu es",
        "c est qui",
        "tu es qui",
    ]

    AGRICULTURE_KEYWORDS = {
        "culture",
        "cultiv",
        "semis",
        "récolte",
        "engrais",
        "pesticide",
        "ravageur",
        "maladie",
        "sol",
        "irrigation",
        "climat",
        "rendement",
        "plant",
        "graine",
        "mil",
        "maïs",
        "riz",
        "arachide",
        "sorgho",
        "coton",
        "café",
        "cacao",
        "élevage",
        "bétail",
        "agriculture",
        "farming",
        "crop",
        "pest",
        "disease",
        "soil",
        "seed",
        "harvest",
        "fertilizer",
        "yield",
        "production",
        "farm",
        "agricultural",
        "traitement",
        "protection",
        "variété",
        "semence",
        "technique",
        "méthode",
        "pratique",
        "produit agricole",
        "légume",
        "fruit",
        "céréale",
        "tubercule",
        "plantation",
        "récolter",
        "planter",
        "arroser",
        "désherber",
        "pulvériser",
        "fumier",
        "compost",
        "NPK",
        "urée",
        "phosphate",
        "potasse",
    }

    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Similarité entre deux chaînes"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    @staticmethod
    def fuzzy_match(question: str, keywords: list, threshold: float = 0.75) -> bool:
        """Fuzzy matching robuste"""
        question_lower = question.lower()
        words = re.findall(r"\b\w+\b", question_lower)

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in question_lower:
                return True
            for word in words:
                if IntentDetector.similarity(word, keyword_lower) >= threshold:
                    return True
        return False

    @staticmethod
    def detect_intent(question: str) -> Tuple[IntentType, float]:
        """Détecte l'intention avec robustesse - VERSION AMÉLIORÉE"""
        question_lower = question.lower().strip()

        if not question_lower or len(question_lower) < 2:
            return (IntentType.OUT_OF_DOMAIN, 0.5)

        # ✅ PRIORITÉ 1 : AGRICULTURE (vérifier EN PREMIER)
        words = re.findall(r"\b\w+\b", question_lower)

        # Termes agricoles clés
        key_ag_terms = [
            "culture",
            "cultiver",
            "engrais",
            "sol",
            "semis",
            "récolte",
            "plante",
            "plantation",
            "agricole",
            "agriculture",
            "farming",
            "mil",
            "maïs",
            "riz",
            "sorgho",
            "arachide",
            "coton",
            "ravageur",
            "maladie",
            "traitement",
            "irrigation",
            "fertilisation",
            "pesticide",
            "fumier",
            "compost",
            "préparer",
        ]

        for term in key_ag_terms:
            if term in question_lower:
                return (IntentType.AGRICULTURE, 0.9)

        # Compter mots agricoles
        ag_matches = 0
        for word in words:
            for ag_keyword in IntentDetector.AGRICULTURE_KEYWORDS:
                if ag_keyword in word or word in ag_keyword:
                    ag_matches += 1
                    break

        ag_score = ag_matches / len(words) if words else 0
        if ag_score > 0.15:
            return (IntentType.AGRICULTURE, max(ag_score, 0.7))

        # ✅ PRIORITÉ 2 : SALUTATIONS (seulement si court)
        if len(question_lower.split()) <= 3 and ag_score == 0:
            if IntentDetector.fuzzy_match(
                question, IntentDetector.GREETING_KEYWORDS, threshold=0.70
            ):
                return (IntentType.GREETING, 0.95)

        # ✅ PRIORITÉ 3 : REMERCIEMENTS
        if len(question_lower.split()) <= 4 and ag_score == 0:
            if IntentDetector.fuzzy_match(
                question, IntentDetector.THANKS_KEYWORDS, threshold=0.70
            ):
                return (IntentType.THANKS, 0.95)

        # ✅ PRIORITÉ 4 : AUTO-DESCRIPTION (très spécifique)
        if ag_score == 0:
            self_patterns = [
                r"\bqui es-tu\b",
                r"\bwho are you\b",
                r"\bton nom\b",
                r"\bprésente-toi\b",
            ]
            for pattern in self_patterns:
                if re.search(pattern, question_lower):
                    return (IntentType.SELF_DESCRIPTION, 0.95)

        # Par défaut : agriculture (mode permissif)
        return (IntentType.AGRICULTURE, 0.65)


# RÉPONSES PRÉDÉFINIES
PREDEFINED_RESPONSES = {
    IntentType.GREETING: {
        "reponse": "Bonjour ! 👋 Bienvenue sur AgroConsulting, votre assistant agricole intelligent. Je suis ici pour répondre à vos questions sur l'agriculture au Burkina Faso et d'Afrique de l'Ouest. Comment puis-je vous aider ?",
    },
    IntentType.THANKS: {
        "reponse": "De rien ! 😊 C'est un plaisir de vous aider. Si vous avez d'autres questions sur l'agriculture, n'hésitez pas à les poser. Je suis là pour vous !",
    },
    IntentType.SELF_DESCRIPTION: {
        "reponse": """Je suis AgroConsulting 🤖, un assistant agricole basé sur l'IA.

**🤖 Qui je suis :**
- Assistant de questions-réponses spécialisé en agriculture
- Basé sur la technologie RAG (Retrieval-Augmented Generation)
- Alimenté par une base de connaissances agricoles du Burkina Faso et d'Afrique de l'Ouest

**📚 Ce que je sais faire :**
- Répondre à des questions sur la culture de plantes (mil, maïs, riz, arachide, sorgho, coton, café, cacao, etc.)
- Fournir des conseils sur les engrais, pesticides et techniques agricoles
- Identifier et prévenir les ravageurs et maladies des cultures
- Expliquer les bonnes pratiques agricoles adaptées au climat
- Citer les sources de mes informations

**🌍 Ma spécialisation :**
- Contexte agricole du Burkina Faso et Afrique de l'Ouest
- Pratiques adaptées au climat semi-aride
- Solutions utilisant des ressources locales

**⚙️ Comment je fonctionne :**
1. Je cherche les documents pertinents dans ma base
2. J'analyse le contenu pour votre question
3. Je génère une réponse basée sur les informations
4. Je cite les sources utilisées

**⚠️ Mes limites :**
- Je ne peux répondre que sur les sujets agricoles
- Si confiance < 60%, je vous le dis honnêtement
- Pour conseil vétérinaire spécialisé, consultez un expert

Je suis toujours là pour vous aider ! 🌾""",
    },
    IntentType.OUT_OF_DOMAIN: {
        "reponse": "Je suis spécialisé uniquement dans l'agriculture. 🌾 Votre question ne semble pas être en rapport avec ce domaine. Pourriez-vous me poser une question sur l'agriculture ?\n\nExemples :\n• Comment cultiver le maïs ?\n• Quels engrais pour le mil ?\n• Comment lutter contre les ravageurs du riz ?\n• Bonnes pratiques agricoles ?\n\nJe serais ravi de vous aider !",
    },
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Question agricole en français",
    )
    max_results: int = Field(default=3, ge=1, le=10)
    template: str = Field(default="standard")
    verbose: bool = Field(default=False)

    @field_validator("template")
    @classmethod
    def validate_template(cls, v):
        valid = ["standard", "concise", "detailed"]
        if v.lower() not in valid:
            raise ValueError(f"Template doit être parmi {valid}")
        return v.lower()


class SourceInfo(BaseModel):
    titre: str
    source: str
    organisme: str
    pertinence: float = Field(ge=0, le=1)
    # Ajout de champs pour permettre au frontend d'accéder directement aux
    # métadonnées du corpus : source_institution (nom de l'organisme) et
    # source_url (lien vers le document)
    source_institution: Optional[str] = None
    source_url: Optional[str] = None


class AskResponse(BaseModel):
    success: bool
    question: str
    reponse: str
    sources: List[SourceInfo]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, Any]
    uptime: float


class SystemInfoResponse(BaseModel):
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
    """API REST Production-Ready pour RAG Agricole"""

    VERSION = "1.0.0"

    def __init__(
        self,
        use_faiss: bool = True,
        enable_cache: bool = True,
        enable_rate_limit: bool = True,
    ):
        self.use_faiss = use_faiss
        self.enable_cache = enable_cache
        self.enable_rate_limit = enable_rate_limit

        self.app = FastAPI(
            title="API RAG Agricole Burkina Faso",
            description="API REST pour questions-réponses agricoles via RAG",
            version=self.VERSION,
            docs_url="/docs",
            redoc_url="/redoc",
        )

        if self.enable_rate_limit:
            self.limiter = Limiter(key_func=get_remote_address)
            self.app.state.limiter = self.limiter
            self.app.add_exception_handler(
                RateLimitExceeded, _rate_limit_exceeded_handler
            )

        self.embedding_model = None
        self.vector_store = None
        self.llm_handler = None
        self.startup_time = datetime.now()
        self.request_count = 0
        self.cache_hits = 0

        self._setup_middleware()
        self._setup_routes()
        self._setup_events()

        # Load external sources index (data/sources.csv) to help resolve local filenames
        try:
            self._sources_index = self._load_sources_index()
        except Exception:
            self._sources_index = []

        self._response_cache: Dict[str, Tuple[AskResponse, float]] = {}
        self.cache_ttl = 3600

    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s"
            )
            return response

    def _normalize_corpus(
        self, corpus: List[Dict], text_field: str = None
    ) -> List[Dict]:
        """Normalise le corpus"""
        if not corpus:
            return []

        if text_field is None:
            for field in ["text", "contenu", "texte", "content"]:
                if field in corpus[0] and corpus[0].get(field):
                    text_field = field
                    break

        if text_field is None:
            for key, value in corpus[0].items():
                if isinstance(value, str) and len(value) > 50:
                    text_field = key
                    break

        normalized = []
        for i, doc in enumerate(corpus):
            contenu = ""
            if text_field and text_field in doc:
                contenu = doc[text_field] or ""
            if not contenu:
                contenu = (
                    doc.get("contenu")
                    or doc.get("text")
                    or doc.get("texte")
                    or doc.get("content")
                    or ""
                )

            normalized_doc = {
                "id": doc.get("id") or doc.get("chunk_id") or f"doc_{i}",
                "titre": doc.get("titre")
                or doc.get("title")
                or doc.get("source", "Document")[:100],
                "contenu": contenu,
                "source": doc.get("source")
                or doc.get("source_institution")
                or "Unknown",
                "organisme": doc.get("organisme")
                or doc.get("source_institution")
                or "Unknown",
                "type": doc.get("type") or "general",
            }

            if "chunk_index" in doc:
                normalized_doc["chunk_index"] = doc["chunk_index"]
            if "total_chunks" in doc:
                normalized_doc["total_chunks"] = doc["total_chunks"]
            if "source_url" in doc:
                normalized_doc["source_url"] = doc["source_url"]

            normalized.append(normalized_doc)

        return normalized

    def _load_sources_index(self) -> List[Dict[str, str]]:
        """Charge data/sources.csv (delimiter ';') et renvoie une liste de dicts
        Chaque entrée contient: id, url, source
        """
        from pathlib import Path

        path = Path("./data/sources.csv")
        if not path.exists():
            return []

        index = []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter=";")
                for row in reader:
                    index.append(
                        {
                            "id": row.get("id"),
                            "url": row.get("url"),
                            "source": row.get("source"),
                        }
                    )
        except Exception:
            logger.exception("Erreur lors du chargement de data/sources.csv")

        return index

    def _resolve_source_url(
        self, raw_url: Optional[str], metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Essaie de normaliser/resoudre un source_url non absolu.

        Strategie (best-effort):
        1. Si raw_url commence par http(s) -> retourne tel quel
        2. Si DOCS_BASE_URL present dans env -> préfixe raw_url
        3. Cherche dans self._sources_index par filename / substring
        4. Cherche par titre/source depuis metadata
        5. Si rien -> retourne raw_url (au moins on ne perd pas l'info)
        """
        if not raw_url and not metadata:
            return None

        # If already absolute
        if (
            raw_url
            and isinstance(raw_url, str)
            and raw_url.lower().startswith(("http://", "https://"))
        ):
            return raw_url

        # Try environment-provided base
        docs_base = os.getenv("DOCS_BASE_URL")
        if raw_url and docs_base:
            candidate = docs_base.rstrip("/") + "/" + raw_url.lstrip("./\/")
            # If it looks valid (starts with http), return it
            if candidate.startswith("http"):
                return candidate

        # Prepare filename and simple normalizations
        filename = None
        if raw_url and isinstance(raw_url, str):
            filename = os.path.basename(raw_url.split("?")[0].split("#")[0])

        # Search index for a row that contains the filename or raw_url fragment
        for row in getattr(self, "_sources_index", []) or []:
            row_url = row.get("url") or ""
            try:
                if raw_url and raw_url in row_url:
                    return row_url
                if filename and filename and row_url.endswith(filename):
                    return row_url
                if filename and filename and filename in row_url:
                    return row_url
            except Exception:
                continue

        # Try matching using metadata title or source
        if metadata:
            titre = (metadata.get("titre") or metadata.get("title") or "").lower()
            src = (
                metadata.get("source") or metadata.get("source_institution") or ""
            ).lower()
            for row in getattr(self, "_sources_index", []) or []:
                row_url = (row.get("url") or "").lower()
                row_src = (row.get("source") or "").lower()
                if titre and any(
                    tok in row_url for tok in titre.split() if len(tok) > 3
                ):
                    return row.get("url")
                if src and src in row_src:
                    return row.get("url")

        # Nothing found -> return the raw_url as-is (frontend will decide linkability)
        return raw_url

    def _setup_events(self):
        """Configure événements startup/shutdown"""

        @self.app.on_event("startup")
        async def startup_event():
            from dotenv import load_dotenv

            load_dotenv()

            logger.info("=" * 70)
            logger.info("[LAUNCH] DEMARRAGE API RAG AGRICOLE BURKINA FASO")
            logger.info("=" * 70)

            try:
                logger.info("[STATS] Chargement modele embeddings...")
                self.embedding_model = EmbeddingPipeline("./data/corpus.json")
                self.embedding_model.initialize_embedding_model()
                logger.info(f"[SUCCESS] Embeddings: {self.embedding_model.model_name}")

                logger.info("[DOCS] Chargement vector store (FAISS)...")
                self.vector_store = FAISSVectorStore("./data/faiss_db")

                from pathlib import Path
                import numpy as np
                import json

                if self.use_faiss:
                    index_exists = (
                        Path("./data/faiss_db/faiss_index.index").exists()
                        and Path("./data/faiss_db/corpus_data.pkl").exists()
                    )
                else:
                    index_exists = Path("./data/chroma_db/corpus_mapping.pkl").exists()

                embeddings_path = "./data/embeddings.npy"
                embeddings_exist = Path(embeddings_path).exists()

                if index_exists:
                    logger.info("[LOAD] Index existant detecte, chargement...")
                    success = self.vector_store.load()
                    if success:
                        stats = self.vector_store.get_statistics()
                        logger.info(
                            f"[SUCCESS] Vector store charge: {stats.get('total_documents', 0)} documents"
                        )

                elif embeddings_exist:
                    logger.info(
                        "[REBUILD] Embeddings trouves, reconstruction de l'index..."
                    )
                    with open("./data/corpus.json", "r", encoding="utf-8") as f:
                        corpus_raw = json.load(f)

                    corpus = self._normalize_corpus(corpus_raw)
                    embeddings = np.load(embeddings_path)
                    self.vector_store.create_index(corpus, embeddings)
                    logger.info(f"[SUCCESS] Index reconstruit: {len(corpus)} documents")

                else:
                    logger.warning("[SETUP] Premier demarrage - Creation de l'index...")
                    with open("./data/corpus.json", "r", encoding="utf-8") as f:
                        corpus = json.load(f)

                    logger.info(f"[SUCCESS] Corpus charge: {len(corpus)} documents")

                    text_field = None
                    for field in ["contenu", "content", "text", "texte"]:
                        if field in corpus[0]:
                            text_field = field
                            break

                    texts = [
                        doc.get(text_field, "") for doc in corpus if doc.get(text_field)
                    ]
                    embeddings = self.embedding_model.embedding_model.encode(
                        texts, batch_size=32, show_progress_bar=True
                    )

                    np.save(embeddings_path, embeddings)
                    corpus_normalized = self._normalize_corpus(
                        corpus, text_field=text_field
                    )
                    self.vector_store.create_index(corpus_normalized, embeddings)
                    logger.info(
                        f"[SUCCESS] Index cree: {len(corpus_normalized)} documents"
                    )

                logger.info("[LLM] Initialisation LLM Handler...")
                self.llm_handler = LLMHandler(
                    backend=LLMBackend.OLLAMA,
                    ollama_model="llama3.2:3b",
                    generation_config=GenerationConfig(
                        temperature=0.1,
                        max_tokens=200,
                        repeat_penalty=1.2,
                        top_p=0.9,
                        top_k=40,
                    ),
                    enable_cache=self.enable_cache,
                )

                health = self.llm_handler.health_check()
                logger.info(f"[SUCCESS] LLM Backend: {health['active_backend']}")
                logger.info("=" * 70)
                logger.info("[SUCCESS] API PRETE - Tous les composants initialises")
                logger.info("=" * 70)
                # reload sources index on startup (best-effort)
                try:
                    self._sources_index = self._load_sources_index()
                    logger.info(
                        f"[SOURCES] Indexed {len(self._sources_index)} external sources"
                    )
                except Exception:
                    logger.warning("[SOURCES] Impossible de charger data/sources.csv")

            except Exception as e:
                logger.error(f"[ERROR] ECHEC INITIALISATION: {e}")
                import traceback

                traceback.print_exc()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            logger.info("[STOP] Arret API - Nettoyage ressources...")
            uptime = (datetime.now() - self.startup_time).total_seconds()
            logger.info(f"[STATS] Uptime: {uptime:.1f}s")
            logger.info(f"[STATS] Requetes totales: {self.request_count}")

    def _setup_routes(self):
        """Définit les routes"""

        @self.app.get("/", tags=["System"])
        async def root():
            return {
                "message": "API RAG Agricole Burkina Faso",
                "version": self.VERSION,
                "status": "operational",
                "documentation": "/docs",
            }

        if self.enable_rate_limit:

            @self.app.post("/ask", response_model=AskResponse, tags=["Q&A"])
            @self.limiter.limit("30/minute")
            async def ask_question(request: Request, ask_request: AskRequest):
                return await self._handle_ask(ask_request)
        else:

            @self.app.post("/ask", response_model=AskResponse, tags=["Q&A"])
            async def ask_question(ask_request: AskRequest):
                return await self._handle_ask(ask_request)

        @self.app.get("/health", response_model=HealthResponse, tags=["System"])
        async def health_check():
            try:
                components_status = {}

                if self.embedding_model and self.embedding_model.embedding_model:
                    try:
                        dimension = self.embedding_model.embedding_model.get_sentence_embedding_dimension()
                        components_status["embedding_model"] = {
                            "status": "healthy",
                            "model": self.embedding_model.model_name,
                            "dimension": dimension,
                        }
                    except Exception as e:
                        components_status["embedding_model"] = {
                            "status": "error",
                            "error": str(e),
                        }
                else:
                    components_status["embedding_model"] = {"status": "not_initialized"}

                if self.vector_store:
                    stats = self.vector_store.get_statistics()
                    components_status["vector_store"] = {
                        "status": "healthy",
                        "type": "FAISS" if self.use_faiss else "ChromaDB",
                        "documents": stats.get("total_documents", 0),
                    }
                else:
                    components_status["vector_store"] = {"status": "not_initialized"}

                if self.llm_handler:
                    llm_health = self.llm_handler.health_check()
                    components_status["llm_handler"] = {
                        "status": "healthy"
                        if llm_health["active_backend"]
                        else "degraded",
                        "backend": llm_health["active_backend"],
                    }
                else:
                    components_status["llm_handler"] = {"status": "not_initialized"}

                all_healthy = all(
                    comp.get("status") == "healthy"
                    for comp in components_status.values()
                )
                overall_status = "healthy" if all_healthy else "degraded"
                uptime = (datetime.now() - self.startup_time).total_seconds()

                return {
                    "status": overall_status,
                    "timestamp": datetime.now().isoformat(),
                    "version": self.VERSION,
                    "components": components_status,
                    "uptime": uptime,
                }
            except Exception as e:
                logger.error(f"Erreur health check: {e}")
                return {
                    "status": "unhealthy",
                    "timestamp": datetime.now().isoformat(),
                    "version": self.VERSION,
                    "components": {"error": str(e)},
                    "uptime": 0,
                }

        @self.app.get(
            "/system/info", response_model=SystemInfoResponse, tags=["System"]
        )
        async def system_info():
            try:
                uptime = (datetime.now() - self.startup_time).total_seconds()
                llm_stats = {}
                if self.llm_handler:
                    llm_stats = self.llm_handler.get_statistics()

                return {
                    "system_name": "API RAG Agricole Burkina Faso",
                    "version": self.VERSION,
                    "description": "Systeme de questions-reponses base sur RAG pour l'agriculture burkinabe",
                    "uptime": uptime,
                    "components": {
                        "embedding_model": self.embedding_model.model_name
                        if self.embedding_model
                        else None,
                        "vector_store": "FAISS" if self.use_faiss else "ChromaDB",
                        "llm_backend": self.llm_handler.active_backend.value
                        if self.llm_handler
                        else None,
                        "cache_enabled": self.enable_cache,
                        "rate_limit_enabled": self.enable_rate_limit,
                    },
                    "endpoints": [
                        {
                            "method": "POST",
                            "path": "/ask",
                            "description": "Poser une question",
                        },
                        {
                            "method": "GET",
                            "path": "/health",
                            "description": "Verifier sante",
                        },
                        {
                            "method": "GET",
                            "path": "/system/info",
                            "description": "Informations systeme",
                        },
                    ],
                    "statistics": {
                        "total_requests": self.request_count,
                        "cache_hits": self.cache_hits,
                        **llm_stats,
                    },
                }
            except Exception as e:
                logger.error(f"Erreur system info: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _handle_ask(self, ask_request: AskRequest) -> AskResponse:
        """Traite une question utilisateur"""
        try:
            self.request_count += 1
            start_time = time.time()

            logger.info(f"[Q] Question recue: {ask_request.question}")

            # Détecter l'intention
            intent, confidence = IntentDetector.detect_intent(ask_request.question)
            logger.info(f"[INTENT] Type: {intent.value}, Confiance: {confidence:.2f}")

            # Traiter intentions spéciales
            if intent != IntentType.AGRICULTURE:
                processing_time = time.time() - start_time
                response_text = PREDEFINED_RESPONSES[intent]["reponse"]

                return AskResponse(
                    success=True,
                    question=ask_request.question,
                    reponse=response_text,
                    sources=[],
                    metadata={
                        "backend": "predefined_response",
                        "model": "intent_detector",
                        "generation_time": 0,
                        "tokens_generated": 0,
                        "tokens_per_second": 0,
                        "documents_used": 0,
                        "processing_time": processing_time,
                        "template": ask_request.template,
                        "cached": False,
                        "intent_type": intent.value,
                        "intent_confidence": confidence,
                    },
                )

            # Question agricole
            if not self._is_system_ready():
                raise HTTPException(status_code=503, detail="Systeme non initialise")

            # Check cache
            if self.enable_cache:
                cache_key = self._get_cache_key(
                    ask_request.question, ask_request.max_results
                )
                cached_response = self._get_from_cache(cache_key)
                if cached_response:
                    self.cache_hits += 1
                    logger.info("[CACHE] Reponse depuis cache")
                    return cached_response

            # Embeddings
            question_embedding = self.embedding_model.embedding_model.encode(
                [ask_request.question]
            )
            if len(question_embedding.shape) > 1:
                question_embedding = question_embedding[0]

            # Recherche
            search_results = self.vector_store.search(
                query_embedding=question_embedding, k=ask_request.max_results
            )

            if not search_results:
                logger.warning("[WARNING] Aucun document pertinent trouve")

            # Vérifier confiance
            avg_score = (
                sum(s.similarity_score for s in search_results) / len(search_results)
                if search_results
                else 0
            )

            logger.info(f"[CONFIDENCE] Score moyen: {avg_score:.3f}")

            if avg_score < 0.6:
                logger.warning(f"[LOW_CONFIDENCE] Score {avg_score:.3f} < 0.6 - Rejet")
                processing_time = time.time() - start_time
                return AskResponse(
                    success=False,
                    question=ask_request.question,
                    reponse="Je n'ai pas d'information fiable sur ce sujet dans ma base de connaissances. Veuillez essayer une autre question ou consulter un expert agricole.",
                    sources=[],
                    metadata={
                        "backend": "N/A",
                        "model": "N/A",
                        "generation_time": 0,
                        "tokens_generated": 0,
                        "tokens_per_second": 0,
                        "documents_used": 0,
                        "processing_time": processing_time,
                        "template": ask_request.template,
                        "cached": False,
                        "confidence_score": avg_score,
                        "rejection_reason": "Confiance insuffisante (< 0.6)",
                        "intent_type": intent.value,
                    },
                )

            # Préparer contexte
            context_docs = []
            sources_info = []

            for result in search_results:
                context_docs.append(
                    {"text": result.document_text, "metadata": result.metadata}
                )

                # Normalize source_url so frontend can show clickable links when possible
                raw_url = (
                    result.metadata.get("source_url")
                    or result.metadata.get("url")
                    or None
                )
                resolved_url = self._resolve_source_url(raw_url, result.metadata)

                sources_info.append(
                    SourceInfo(
                        titre=result.metadata.get("titre", "Document inconnu"),
                        source=result.metadata.get("source", "Source inconnue"),
                        # source_institution dans le corpus correspond à organisme
                        organisme=(
                            result.metadata.get("organisme")
                            or result.metadata.get("source_institution")
                            or "N/A"
                        ),
                        pertinence=float(result.similarity_score),
                        source_institution=(
                            result.metadata.get("source_institution")
                            or result.metadata.get("organisme")
                            or None
                        ),
                        source_url=resolved_url,
                    )
                )

                if ask_request.verbose:
                    logger.info(
                        f"   - {result.metadata.get('titre', 'Doc')} (score: {result.similarity_score:.3f})"
                    )

            # Template mapping
            template_map = {
                "standard": PromptTemplate.STANDARD,
                "concise": PromptTemplate.CONCISE,
                "detailed": PromptTemplate.DETAILED,
            }
            template = template_map.get(ask_request.template, PromptTemplate.STANDARD)

            # Génération LLM
            if ask_request.verbose:
                logger.info("[LLM] Generation reponse...")

            llm_response = self.llm_handler.generate_answer(
                ask_request.question,
                context_docs,
                template=template,
            )

            processing_time = time.time() - start_time

            # Construire réponse
            response = AskResponse(
                success=llm_response.success,
                question=ask_request.question,
                reponse=llm_response.text,
                sources=sources_info,
                metadata={
                    "backend": llm_response.backend,
                    "model": llm_response.model,
                    "generation_time": llm_response.generation_time,
                    "tokens_generated": llm_response.tokens_generated,
                    "tokens_per_second": llm_response.tokens_per_second,
                    "documents_used": len(search_results),
                    "processing_time": processing_time,
                    "template": ask_request.template,
                    "cached": False,
                    "confidence_score": avg_score,
                    "intent_type": intent.value,
                },
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
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

    def _is_system_ready(self) -> bool:
        """Vérifie que tous les composants sont prêts"""
        return all(
            [
                self.embedding_model is not None,
                self.vector_store is not None,
                self.llm_handler is not None,
            ]
        )

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
        """Sauvegarde en cache"""
        self._response_cache[cache_key] = (response, time.time())
        if len(self._response_cache) > 100:
            oldest_key = min(
                self._response_cache.keys(), key=lambda k: self._response_cache[k][1]
            )
            del self._response_cache[oldest_key]


# ============================================================================
# CRÉATION INSTANCE API
# ============================================================================

api_instance = AgriculturalRAGAPI(
    use_faiss=True,
    enable_cache=True,
    enable_rate_limit=True,
)

app = api_instance.app


# ============================================================================
# FONCTION DE DÉMARRAGE
# ============================================================================


def start_server(
    host: str = "0.0.0.0", port: int = 8000, reload: bool = False, workers: int = 1
):
    """Démarre le serveur FastAPI"""
    logger.info(f"[LAUNCH] Demarrage serveur sur {host}:{port}")

    import sys

    if sys.platform == "win32":
        workers = 1
        reload = True

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
        access_log=True,
        loop="asyncio",
    )


if __name__ == "__main__":
    start_server(
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
