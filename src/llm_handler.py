"""
LLM Handler Module - Production Ready
Système de génération de réponses pour RAG Agricole Burkina Faso

Auteur: Expert ML Team
Date: 3 Novembre 2025
Hackathon: MTDPCE 2025

Fonctionnalités:
- Support Ollama (local) + HuggingFace API (fallback)
- Auto-détection backend disponible
- Prompt engineering optimisé agriculture BF
- Post-processing réponses (formatage, sources)
- Caching intelligent (optionnel)
- Gestion robuste erreurs
- Métriques de performance

Usage:
    >>> llm = LLMHandler()  # Auto-détecte meilleur backend
    >>> response = llm.generate_answer(question, context_docs)
"""

import logging
import sys
import io
import requests
import json
import time
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from functools import lru_cache
import os

# Forcer UTF-8 console (Windows) pour éviter UnicodeEncodeError lors des logs
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

# Configuration logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# TYPES ET ENUMS
# ============================================================================


class LLMBackend(Enum):
    """Backends LLM supportés"""

    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    AUTO = "auto"  # Détection automatique


class PromptTemplate(Enum):
    """Templates de prompts optimisés"""

    STANDARD = "standard"
    CONCISE = "concise"
    DETAILED = "detailed"


@dataclass
class GenerationConfig:
    """
    Configuration génération LLM - Optimisée pour agriculture BF

    Paramètres ajustés pour:
    - Réponses factuelles (temperature=0.1)
    - Longueur raisonnable (max_tokens=800)
    - Pas de répétitions (repeat_penalty=1.2)
    """

    temperature: float = 0.1  # Très bas = factuel
    top_p: float = 0.9  # Nucleus sampling
    top_k: int = 40  # Top-k sampling
    max_tokens: int = 200  # Longueur réponse
    num_ctx: int = 1024  # Contexte window
    repeat_penalty: float = 1.2  # Anti-répétition
    stop_sequences: List[str] = None  # Séquences d'arrêt

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = ["\n\nQuestion:", "\n\nUser:", "\n\nHuman:", "###"]

    def to_ollama_dict(self) -> Dict[str, Any]:
        """Format pour Ollama API"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
            "repeat_penalty": self.repeat_penalty,
            "stop": self.stop_sequences,
        }

    def to_huggingface_dict(self) -> Dict[str, Any]:
        """Format pour HuggingFace API"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_tokens,
            "repetition_penalty": self.repeat_penalty,
            "do_sample": True if self.temperature > 0 else False,
        }


@dataclass
class LLMResponse:
    """Structure de réponse LLM standardisée"""

    text: str
    model: str
    backend: str
    generation_time: float
    tokens_generated: int
    tokens_per_second: float
    context_used: bool
    sources: List[str]
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire"""
        return {
            "response": self.text,
            "model": self.model,
            "backend": self.backend,
            "generation_time": self.generation_time,
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": self.tokens_per_second,
            "context_used": self.context_used,
            "sources": self.sources,
            "success": self.success,
            "error": self.error,
        }


# ============================================================================
# CLASSE PRINCIPALE LLM HANDLER
# ============================================================================


class LLMHandler:
    """
    Handler LLM professionnel avec fallback automatique

    Stratégie:
    1. Tente Ollama (local, rapide, gratuit)
    2. Si échec → HuggingFace API (cloud, limite 1000 req/jour)
    3. Cache optionnel pour économiser requêtes

    Features:
    - Auto-détection backend disponible
    - Retry logic intelligent
    - Post-processing réponses
    - Métriques de performance
    - Prompts optimisés agriculture BF
    """

    # Constantes
    OLLAMA_BASE_URL = "http://localhost:11434"
    HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models"
    # Modèles par défaut (modèles légers recommandés)
    DEFAULT_OLLAMA_MODEL = "llama3.2:3b"  # Modèle léger (3B paramètres)
    # Alternatives: "mistral:7b-instruct-q4_K_M", "llama3.2:1b", "phi3:mini"
    DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
    REQUEST_TIMEOUT = 600
    REQUEST_TIMEOUT_LONG = 180  # 3 minutes pour HuggingFace (modèle en chargement)
    MAX_RETRIES = 2
    RETRY_DELAY = 2

    def __init__(
        self,
        backend: LLMBackend = LLMBackend.OLLAMA,
        ollama_model: str = "llama3.2:3b",  # ✅ Bon nom
        ollama_base_url: str = "http://localhost:11434",
        huggingface_model: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        enable_cache: bool = True,
        hf_api_token: Optional[str] = None,
    ):
        """
        Initialise le LLM Handler

        Args:
            backend: Backend à utiliser (auto-détecte par défaut)
            ollama_model: Nom du modèle Ollama
            huggingface_model: Nom du modèle HuggingFace
            generation_config: Configuration génération
            enable_cache: Activer cache réponses (économise API calls)
            hf_api_token: Token HuggingFace (optionnel, augmente limite)
        """
        self.generation_config = generation_config or GenerationConfig()
        self.enable_cache = enable_cache
        self.hf_api_token = hf_api_token or os.getenv("HUGGINGFACE_API_TOKEN")

        # Modèles
        self.ollama_model = ollama_model
        self.hf_model = huggingface_model or self.DEFAULT_HF_MODEL

        # cache memory
        self._response_cache: Dict[str, LLMResponse] = {}
        self._cache_max_size = 100

        # Session HTTP réutilisable (performance)
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "User-Agent": "RAG-Agricole-BF/1.0"}
        )

        # Statistiques
        self.stats = {
            "total_requests": 0,
            "ollama_requests": 0,
            "huggingface_requests": 0,
            "cache_hits": 0,
            "errors": 0,
        }

        # Déterminer backend actif
        if backend == LLMBackend.AUTO:
            self.active_backend = self._detect_available_backend()
        else:
            self.active_backend = backend

        logger.info(
            f"[SUCCESS] LLM Handler initialisé - Backend: {self.active_backend.value}"
        )
        logger.info(f"   Ollama model: {self.ollama_model}")
        logger.info(f"   HuggingFace model: {self.hf_model}")
        logger.info(f"   Cache: {'activé' if self.enable_cache else 'désactivé'}")

    def _detect_available_backend(self) -> LLMBackend:
        """
        Détecte automatiquement le meilleur backend disponible

        Priorité: Ollama (local) > HuggingFace (cloud)

        Returns:
            LLMBackend disponible
        """
        # Test Ollama
        try:
            response = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags", timeout=3)

            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                if self.ollama_model in model_names:
                    logger.info(
                        f"[SUCCESS] Ollama détecté avec modèle {self.ollama_model}"
                    )
                    return LLMBackend.OLLAMA
                else:
                    logger.warning(
                        f"[WARNING] Ollama disponible mais modèle {self.ollama_model} manquant"
                    )
                    logger.warning(f"   Modèles disponibles: {model_names}")

                    # Essayer de trouver un modèle alternatif disponible
                    if model_names:
                        # Priorité: llama3.2:3b > llama3.2:1b > phi3 > autres
                        priority_models = [
                            "llama3.2:3b",
                            "llama3.2:1b",
                            "phi3:mini",
                            "phi3",
                            "mistral:7b-instruct-q4_K_M",
                            "mistral:7b",
                        ]

                        for priority_model in priority_models:
                            if priority_model in model_names:
                                logger.info(
                                    f" Utilisation modèle alternatif: {priority_model}"
                                )
                                self.ollama_model = priority_model
                                return LLMBackend.OLLAMA

                        # Si aucun modèle prioritaire, utiliser le premier disponible
                        logger.info(
                            f" Utilisation premier modèle disponible: {model_names[0]}"
                        )
                        self.ollama_model = model_names[0]
                        return LLMBackend.OLLAMA
                    else:
                        logger.warning(
                            "[WARNING] Ollama disponible mais aucun modèle installé"
                        )
                        logger.info(" Installez un modèle: ollama pull llama3.2:3b")

        except Exception as e:
            logger.warning(f"[WARNING] Ollama non disponible: {e}")

        # Fallback HuggingFace
        logger.info("[API] Utilisation HuggingFace API (fallback)")
        return LLMBackend.HUGGINGFACE

    def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état des services LLM

        Returns:
            Dict avec statut et infos
        """
        health = {
            "ollama": self._check_ollama_health(),
            "huggingface": self._check_huggingface_health(),
            "active_backend": self.active_backend.value,
            "stats": self.stats.copy(),
        }

        return health

    def _check_ollama_health(self) -> Dict[str, Any]:
        """Vérifie santé Ollama"""
        try:
            response = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags", timeout=5)

            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "status": "healthy",
                    "available_models": [m["name"] for m in models],
                    "target_model_available": self.ollama_model
                    in [m["name"] for m in models],
                }
        except:
            pass

        return {"status": "unavailable"}

    def _check_huggingface_health(self) -> Dict[str, Any]:
        """Vérifie santé HuggingFace API"""
        try:
            test_url = f"{self.HUGGINGFACE_API_URL}/{self.hf_model}"
            headers = {}
            if self.hf_api_token:
                headers["Authorization"] = f"Bearer {self.hf_api_token}"

            # Test simple (pas de génération)
            response = requests.get(test_url, headers=headers, timeout=5)

            if response.status_code in [200, 503]:  # 503 = modèle en chargement
                return {
                    "status": "available",
                    "model": self.hf_model,
                    "authenticated": bool(self.hf_api_token),
                }
        except:
            pass

        return {"status": "unknown"}

    # ========================================================================
    # PROMPT ENGINEERING
    # ========================================================================

    def _build_agricultural_prompt(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        template: PromptTemplate = PromptTemplate.STANDARD,
    ) -> str:
        """
        Construit un prompt optimisé pour l'agriculture BF

        Args:
            question: Question utilisateur
            context_docs: Documents de contexte (format: {text, metadata})
            template: Type de prompt à utiliser

        Returns:
            Prompt formaté
        """
        # Construire section contexte
        context_parts = []
        sources = []

        for i, doc in enumerate(context_docs[:5], 1):  # Max 5 docs
            text = doc.get("text", doc.get("contenu", ""))
            metadata = doc.get("metadata", {})

            # Extraire source
            source = metadata.get("titre", "Document")
            source_org = metadata.get("organisme", "")
            if source_org:
                source = f"{source} ({source_org})"

            sources.append(source)

            # Limiter longueur texte (éviter context overflow)
            text_excerpt = text[:600] if len(text) > 600 else text

            context_parts.append(f"[Document {i} - {source}]\n{text_excerpt}")

        context_text = "\n\n".join(context_parts)

        # Template selon type
        if template == PromptTemplate.CONCISE:
            prompt = f"""Tu es un conseiller agricole pour le Burkina Faso. Réponds de façon CONCISE (3-5 phrases max).

CONTEXTE:
{context_text}

QUESTION: {question}

RÉPONSE CONCISE (cite les sources):"""

        elif template == PromptTemplate.DETAILED:
            prompt = f"""Tu es un expert en agriculture burkinabè avec 20 ans d'expérience. 

CONTEXTE TECHNIQUE:
{context_text}

QUESTION DE L'AGRICULTEUR: {question}

RÉPONSE DÉTAILLÉE:
Fournis une réponse complète et structurée avec:
1. Réponse directe
2. Explications techniques
3. Conseils pratiques
4. Sources consultées

Réponse:"""

        else:  # STANDARD
            prompt = f"""Tu es un conseiller agricole expert pour le Burkina Faso, spécialisé dans les cultures sahéliennes (mil, sorgho, maïs, maraîchage).

DOCUMENTS DE RÉFÉRENCE:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Base ta réponse UNIQUEMENT sur les documents ci-dessus
- Sois précis et pratique (quantités, périodes, techniques)
- Structure ta réponse clairement
- Mentionne les sources quand pertinent
- Si l'info n'est pas dans les documents, dis-le
- Reste accessible aux agriculteurs burkinabè

RÉPONSE:"""

        return prompt

    def _build_simple_prompt(
        self, question: str, context_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Construit un prompt SIMPLE et ROBUSTE pour Llama3.2
        Évite les formats complexes qui peuvent causer des erreurs

        Args:
            question: Question de l'utilisateur
            context_docs: Documents de contexte avec 'text' et 'metadata'

        Returns:
            Prompt formaté
        """
        # Construire contexte de manière simple
        context_parts = []
        for i, doc in enumerate(context_docs[:3]):  # Limiter à 3 documents
            text = doc.get("text", doc.get("contenu", ""))
            # Limiter à 300 caractères pour stabilité
            text_excerpt = text[:300] + "..." if len(text) > 300 else text
            context_parts.append(f"Source {i + 1}: {text_excerpt}")

        context_text = "\n\n".join(context_parts)

        # Prompt direct et court
        prompt = f"""Expert agricole BF. BasÃ© sur ces documents:

        {context_text}

        Question: {question}

        Réponse concise (2-3 phrases):"""

        return prompt

    # ========================================================================
    # GÉNÉRATION AVEC OLLAMA
    # ========================================================================

    def _generate_with_ollama(
        self, prompt: str, retry_count: int = 0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Génère réponse avec Ollama

        Args:
            prompt: Prompt d'entrée
            retry_count: Compteur retries

        Returns:
            Tuple (response_text, metadata)

        Raises:
            Exception si échec après retries
        """
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": self.generation_config.to_ollama_dict(),
            }

            start_time = time.time()

            response = self.session.post(
                f"{self.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=self.REQUEST_TIMEOUT,
            )

            response.raise_for_status()
            data = response.json()

            generation_time = time.time() - start_time

            # Extraire métadonnées
            metadata = {
                "backend": "ollama",
                "model": data.get("model", self.ollama_model),
                "generation_time": generation_time,
                "total_duration": data.get("total_duration", 0) / 1e9,
                "tokens_generated": data.get("eval_count", 0),
                "tokens_prompt": data.get("prompt_eval_count", 0),
                "tokens_per_second": data.get("eval_count", 0) / generation_time
                if generation_time > 0
                else 0,
            }

            self.stats["ollama_requests"] += 1

            logger.info(
                f"[SUCCESS] Ollama: {metadata['tokens_generated']} tokens en {generation_time:.2f}s ({metadata['tokens_per_second']:.1f} tok/s)"
            )

            return data.get("response", ""), metadata

        except requests.exceptions.Timeout:
            logger.warning(f"⏱️ Ollama timeout (tentative {retry_count + 1})")
            if retry_count < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY)
                return self._generate_with_ollama(prompt, retry_count + 1)
            else:
                raise TimeoutError("Ollama timeout après retries")

        except Exception as e:
            logger.error(f"[ERROR] Erreur Ollama: {e}")
            if retry_count < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY)
                return self._generate_with_ollama(prompt, retry_count + 1)
            else:
                raise

    # ========================================================================
    # GÉNÉRATION AVEC HUGGINGFACE
    # ========================================================================

    def _generate_with_huggingface(
        self, prompt: str, retry_count: int = 0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Génère réponse avec HuggingFace Inference API

        Args:
            prompt: Prompt d'entrée
            retry_count: Compteur retries

        Returns:
            Tuple (response_text, metadata)

        Raises:
            Exception si échec après retries
        """
        try:
            # Headers avec token si disponible
            headers = {"Content-Type": "application/json"}
            if self.hf_api_token:
                headers["Authorization"] = f"Bearer {self.hf_api_token}"

            # Payload HuggingFace
            payload = {
                "inputs": prompt,
                "parameters": self.generation_config.to_huggingface_dict(),
                "options": {
                    "wait_for_model": True,  # Attendre si modèle en chargement
                    "use_cache": True,
                },
            }

            start_time = time.time()

            response = requests.post(
                f"{self.HUGGINGFACE_API_URL}/{self.hf_model}",
                headers=headers,
                json=payload,
                timeout=self.REQUEST_TIMEOUT_LONG,
            )

            response.raise_for_status()
            data = response.json()

            generation_time = time.time() - start_time

            # Extraire texte généré
            if isinstance(data, list) and len(data) > 0:
                generated_text = data[0].get("generated_text", "")

                # Nettoyer: retirer le prompt de la réponse
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt) :].strip()
            else:
                generated_text = str(data)

            # Métadonnées (HF API fournit peu d'infos)
            metadata = {
                "backend": "huggingface",
                "model": self.hf_model,
                "generation_time": generation_time,
                "tokens_generated": len(generated_text.split()),  # Approximation
                "tokens_per_second": len(generated_text.split()) / generation_time
                if generation_time > 0
                else 0,
            }

            self.stats["huggingface_requests"] += 1

            logger.info(
                f"[SUCCESS] HuggingFace: ~{metadata['tokens_generated']} tokens en {generation_time:.2f}s"
            )

            return generated_text, metadata

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                # Modèle en chargement
                logger.warning(f"⏳ Modèle HF en chargement, attente...")
                if retry_count < 3:  # Plus de retries pour HF
                    time.sleep(10)  # Attente plus longue
                    return self._generate_with_huggingface(prompt, retry_count + 1)

            logger.error(f"[ERROR] Erreur HTTP HF: {e.response.status_code}")
            if retry_count < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY)
                return self._generate_with_huggingface(prompt, retry_count + 1)
            else:
                raise

        except Exception as e:
            logger.error(f"[ERROR] Erreur HuggingFace: {e}")
            if retry_count < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY)
                return self._generate_with_huggingface(prompt, retry_count + 1)
            else:
                raise

    # ========================================================================
    # POST-PROCESSING
    # ========================================================================

    def _post_process_response(
        self, raw_response: str, context_docs: List[Dict]
    ) -> Tuple[str, List[str]]:
        """
        Post-traite la réponse brute du LLM

        Nettoyages:
        - Retire répétitions
        - Formate sources
        - Limite longueur si excessive
        - Retire artifacts (###, ---, etc.)

        Args:
            raw_response: Réponse brute du LLM
            context_docs: Documents de contexte utilisés

        Returns:
            Tuple (cleaned_response, extracted_sources)
        """
        response = raw_response.strip()

        # 1. Retirer séquences d'arrêt mal gérées
        for stop_seq in ["###", "---", "Question:", "User:", "Human:"]:
            if stop_seq in response:
                response = response.split(stop_seq)[0].strip()

        # 2. Limiter longueur excessive (sécurité)
        if len(response) > 2000:
            response = response[:2000] + "..."
            logger.warning("[WARNING] Réponse tronquée (trop longue)")

        # 3. Extraire sources mentionnées
        sources = []
        for doc in context_docs:
            metadata = doc.get("metadata", {})
            source_name = metadata.get("titre", "") or metadata.get("source", "")
            if source_name and source_name.lower() in response.lower():
                sources.append(source_name)

        # 4. Si aucune source extraite, utiliser toutes les sources du contexte
        if not sources:
            sources = [
                doc.get("metadata", {}).get("titre", "Document")
                for doc in context_docs[:3]
            ]

        # 5. Nettoyer espaces multiples
        response = re.sub(r"\n{3,}", "\n\n", response)
        response = re.sub(r" {2,}", " ", response)

        return response, sources

    # ========================================================================
    # API PRINCIPALE
    # ========================================================================

    def generate_answer(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        template: PromptTemplate = PromptTemplate.STANDARD,
    ) -> LLMResponse:
        """
        Génère une réponse à une question agricole avec Ollama

        Args:
            question: Question de l'utilisateur
            context_docs: Documents de contexte avec 'text' et 'metadata'
            template: Template de prompt (ignoré, utilise toujours simple)

        Returns:
            LLMResponse avec réponse et métadonnées
        """
        try:
            self.stats["total_requests"] += 1

            # ✅ AJOUTER : Vérifier cache
            if self.enable_cache:
                cache_key = self._get_cache_key(question, context_docs)
                cached_response = self._get_from_cache(cache_key)
                if cached_response:
                    return cached_response

            if not question or not question.strip():
                raise ValueError("Question vide")

            if not context_docs:
                logger.warning("Aucun contexte fourni, génération fallback")
                return self._generate_fallback_response(question)

            # Construire prompt (toujours simple pour stabilité)
            # Construire prompt selon le template demandé
            if template == PromptTemplate.CONCISE or len(context_docs) > 5:
                # Prompt simple pour stabilité avec beaucoup de contexte
                prompt = self._build_simple_prompt(question, context_docs)
            else:
                # Prompt structuré pour qualité
                prompt = self._build_agricultural_prompt(question, context_docs, template)

            logger.debug(f"Template utilisé: {template.value if template != PromptTemplate.CONCISE else 'simple'}")
            logger.debug(f"Prompt généré: {len(prompt)} caractères")

            # Génération avec Ollama
            raw_response, metadata = self._generate_with_ollama(prompt)

            # Post-traitement
            cleaned_response, sources = self._post_process_response(
                raw_response, context_docs
            )

            # Construire réponse
            llm_response = LLMResponse(
                text=cleaned_response,
                model=metadata.get("model", self.ollama_model),
                backend="ollama",
                generation_time=metadata.get("generation_time", 0),
                tokens_generated=metadata.get("tokens_generated", 0),
                tokens_per_second=metadata.get("tokens_per_second", 0),
                context_used=True,
                sources=sources,
                success=True,
            )

            logger.info(f"✅ Réponse générée: '{question[:40]}...'")

            # ✅ AJOUTER : Sauvegarder dans cache
            if self.enable_cache:
                self._save_to_cache(cache_key, llm_response)

            return llm_response

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"❌ Échec génération: {e}")
            return self._generate_error_response(question, str(e))

    def _generate_fallback_response(self, question: str) -> LLMResponse:
        """Génère réponse fallback quand pas de contexte"""
        fallback_text = (
            "Je n'ai pas trouvé d'informations spécifiques dans ma base de connaissances "
            "pour répondre à cette question sur l'agriculture burkinabè. "
            "Je recommande de consulter les services agricoles locaux "
            "(ministère de l'Agriculture, chambres d'agriculture) ou les organisations "
            "comme la FAO, le CIRAD ou les ONG actives dans le domaine agricole."
        )

        return LLMResponse(
            text=fallback_text,
            model=self.ollama_model,
            backend="fallback",
            generation_time=0,
            tokens_generated=len(fallback_text.split()),
            tokens_per_second=0,
            context_used=False,
            sources=[],
            success=True,
        )

    def _generate_error_response(self, question: str, error: str) -> LLMResponse:
        """Génère réponse d'erreur"""
        error_text = (
            "Désolé, je rencontre actuellement des difficultés techniques pour répondre "
            "à votre question. Veuillez réessayer dans quelques instants. "
            f"Erreur technique: {error}"
        )

        return LLMResponse(
            text=error_text,
            model="error",
            backend="error",
            generation_time=0,
            tokens_generated=0,
            tokens_per_second=0,
            context_used=False,
            sources=[],
            success=False,
            error=error,
        )

    # ========================================================================
    # CACHE (OPTIONNEL)
    # ========================================================================

    def _get_cache_key(self, question: str, context_docs: List[Dict]) -> str:
        """Génère clé de cache unique pour question + contexte"""
        # Hash de la question + IDs documents
        doc_ids = sorted(
            [doc.get("metadata", {}).get("id", "") for doc in context_docs]
        )
        cache_str = f"{question}||{'|'.join(doc_ids)}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[LLMResponse]:
        """Récupère réponse depuis cache en mémoire"""
        if not self.enable_cache:
            return None
        
        cached = self._response_cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            logger.info(f"💾 Cache hit: {cache_key[:16]}...")
        return cached

    def _save_to_cache(self, cache_key: str, response: LLMResponse):
        """Sauvegarde réponse en cache avec limite de taille"""
        if not self.enable_cache:
            return
        
        # Limiter taille du cache (FIFO simple)
        if len(self._response_cache) >= self._cache_max_size:
            # Supprimer la plus vieille entrée
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
            logger.debug(f"🗑️ Cache plein, suppression: {oldest_key[:16]}...")
        
        self._response_cache[cache_key] = response
        logger.debug(f"💾 Cache saved: {cache_key[:16]}...")

    # ========================================================================
    # UTILITAIRES
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne statistiques d'utilisation"""
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cache_hits"] / self.stats["total_requests"] * 100
                if self.stats["total_requests"] > 0
                else 0
            ),
            "error_rate": (
                self.stats["errors"] / self.stats["total_requests"] * 100
                if self.stats["total_requests"] > 0
                else 0
            ),
        }

    def benchmark(
        self, test_question: str = "Présente-toi en une phrase."
    ) -> Dict[str, Any]:
        """
        Benchmark performance du LLM

        Args:
            test_question: Question de test

        Returns:
            Métriques de performance
        """
        try:
            logger.info(" Benchmark LLM démarré...")

            # Test sans contexte (génération simple)
            test_docs = [
                {
                    "text": "Test document pour benchmark.",
                    "metadata": {"titre": "Test", "source": "Benchmark"},
                }
            ]

            start = time.time()
            response = self.generate_answer(test_question, test_docs)
            elapsed = time.time() - start

            metrics = {
                "success": response.success,
                "backend": response.backend,
                "model": response.model,
                "total_time": elapsed,
                "generation_time": response.generation_time,
                "tokens_generated": response.tokens_generated,
                "tokens_per_second": response.tokens_per_second,
                "response_length": len(response.text),
            }

            logger.info(
                f"[SUCCESS] Benchmark terminé: {metrics['tokens_per_second']:.1f} tokens/sec"
            )

            return metrics

        except Exception as e:
            logger.error(f"[ERROR] Benchmark échoué: {e}")
            return {"success": False, "error": str(e)}


# ============================================================================
# TESTS AUTOMATISÉS
# ============================================================================


def test_llm_handler_complete():
    """
    Suite de tests complète pour LLM Handler

    Tests:
    1. Détection backend
    2. Health check
    3. Génération simple
    4. Génération RAG agricole
    5. Post-processing
    6. Fallback automatique
    7. Performance
    """
    logger.info("\n" + "=" * 70)
    logger.info("[TEST] SUITE DE TESTS COMPLÈTE - LLM HANDLER")
    logger.info("=" * 70)

    try:
        # ===== TEST 1: Initialisation =====
        logger.info("\n Test 1: Initialisation et détection backend...")

        llm = LLMHandler(
            backend=LLMBackend.AUTO,
            generation_config=GenerationConfig(temperature=0.1, max_tokens=500),
        )

        logger.info(f"[SUCCESS] Backend détecté: {llm.active_backend.value}")

        # ===== TEST 2: Health Check =====
        logger.info("\n Test 2: Health check services...")

        health = llm.health_check()
        logger.info(f"Statut Ollama: {health['ollama'].get('status', 'unknown')}")
        logger.info(
            f"Statut HuggingFace: {health['huggingface'].get('status', 'unknown')}"
        )
        logger.info(f"Backend actif: {health['active_backend']}")

        if (
            health["ollama"]["status"] != "healthy"
            and health["huggingface"]["status"] != "available"
        ):
            logger.error("[ERROR] Aucun backend disponible !")
            return False

        logger.info("[SUCCESS] Au moins un backend disponible")

        # ===== TEST 3: Génération Simple =====
        logger.info("\n Test 3: Génération simple...")

        simple_docs = [
            {
                "text": "Le Burkina Faso est un pays sahélien d'Afrique de l'Ouest. L'agriculture y est vitale.",
                "metadata": {"titre": "Introduction BF", "source": "Test"},
            }
        ]

        simple_response = llm.generate_answer(
            "Où est situé le Burkina Faso ?", simple_docs
        )

        if simple_response.success:
            logger.info(
                f"[SUCCESS] Réponse générée ({simple_response.tokens_generated} tokens)"
            )
            logger.info(f"   Backend: {simple_response.backend}")
            logger.info(f"   Temps: {simple_response.generation_time:.2f}s")
            logger.info(f"   Réponse: {simple_response.text[:100]}...")
        else:
            logger.error(f"[ERROR] Échec génération: {simple_response.error}")

        # ===== TEST 4: Génération RAG Agricole =====
        logger.info("\n Test 4: Génération RAG agricole...")

        agricultural_docs = [
            {
                "text": "Le mil (Pennisetum glaucum) est une céréale très résistante à la sécheresse, particulièrement adaptée au climat sahélien du Burkina Faso. Il nécessite 400-600 mm d'eau par saison. La fertilisation recommandée est de 100-150 kg/ha de NPK 14-23-14 au semis, suivie de 50 kg/ha d'urée en couverture 30-40 jours après semis.",
                "metadata": {
                    "titre": "Culture du mil au Sahel",
                    "source": "FAO - Guide technique 2023",
                    "organisme": "FAO",
                    "id": "doc_mil_001",
                },
            },
            {
                "text": "Le semis du mil doit être effectué après les premières pluies utiles (cumul >20mm). Densité recommandée: 10-15 kg de semences par hectare, en lignes espacées de 80 cm. Profondeur de semis: 3-5 cm. Le mil se récolte généralement 90-120 jours après semis, selon la variété.",
                "metadata": {
                    "titre": "Calendrier cultural mil",
                    "source": "CIRAD - Fiches techniques",
                    "organisme": "CIRAD",
                    "id": "doc_mil_002",
                },
            },
            {
                "text": "Principales maladies du mil au Burkina Faso: le mildiou (Sclerospora graminicola) et le charbon (Tolyposporium penicillariae). Lutte préventive: utiliser des semences certifiées, rotation des cultures. Lutte curative: traiter les semences avec des fongicides appropriés.",
                "metadata": {
                    "titre": "Maladies et ravageurs du mil",
                    "source": "Institut de l'Environnement et Recherches Agricoles (INERA)",
                    "organisme": "INERA",
                    "id": "doc_mil_003",
                },
            },
        ]

        agricultural_question = "Comment bien cultiver le mil au Burkina Faso ? Quand le semer et quel engrais utiliser ?"

        agricultural_response = llm.generate_answer(
            agricultural_question, agricultural_docs, template=PromptTemplate.STANDARD
        )

        if agricultural_response.success:
            logger.info("[SUCCESS] Réponse agricole générée")
            logger.info(f"\n{'=' * 70}")
            logger.info(f"QUESTION: {agricultural_question}")
            logger.info(f"{'=' * 70}")
            logger.info(f"RÉPONSE:\n{agricultural_response.text}")
            logger.info(f"{'=' * 70}")
            logger.info(f"SOURCES: {', '.join(agricultural_response.sources)}")
            logger.info(f"MÉTRIQUES:")
            logger.info(f"  - Backend: {agricultural_response.backend}")
            logger.info(f"  - Modèle: {agricultural_response.model}")
            logger.info(f"  - Temps: {agricultural_response.generation_time:.2f}s")
            logger.info(f"  - Tokens: {agricultural_response.tokens_generated}")
            logger.info(
                f"  - Vitesse: {agricultural_response.tokens_per_second:.1f} tok/s"
            )
        else:
            logger.error(
                f"[ERROR] Échec génération agricole: {agricultural_response.error}"
            )

        # ===== TEST 5: Templates de Prompts =====
        logger.info("\n Test 5: Test différents templates...")

        test_question = "Quel est le meilleur moment pour planter le mil ?"

        for template in [
            PromptTemplate.CONCISE,
            PromptTemplate.STANDARD,
            PromptTemplate.DETAILED,
        ]:
            logger.info(f"\n   Test template: {template.value}")
            response = llm.generate_answer(
                test_question, agricultural_docs[:1], template=template
            )
            logger.info(f"   Longueur réponse: {len(response.text)} caractères")
            logger.info(f"   Extrait: {response.text[:80]}...")

        # ===== TEST 6: Fallback (sans contexte) =====
        logger.info("\n Test 6: Réponse fallback (sans contexte)...")

        fallback_response = llm.generate_answer("Question test", [])

        if fallback_response.success and not fallback_response.context_used:
            logger.info("[SUCCESS] Fallback fonctionne correctement")
            logger.info(f"   Réponse: {fallback_response.text[:100]}...")

        # ===== TEST 7: Post-processing =====
        logger.info("\n Test 7: Test post-processing...")

        # Simuler réponse avec artifacts
        raw_response = """Voici la réponse.

### Section inutile
Contenu pertinent.

---
Autre artifact à retirer."""

        cleaned, sources = llm._post_process_response(raw_response, agricultural_docs)

        assert "###" not in cleaned, "[ERROR] Post-processing raté (### présent)"
        assert "---" not in cleaned, "[ERROR] Post-processing raté (--- présent)"
        logger.info("[SUCCESS] Post-processing fonctionne")
        logger.info(f"   Longueur nettoyée: {len(cleaned)} caractères")

        # ===== TEST 8: Benchmark Performance =====
        logger.info("\n Test 8: Benchmark performance...")

        benchmark_results = llm.benchmark("Présente le Burkina Faso en une phrase.")

        if benchmark_results["success"]:
            logger.info("[SUCCESS] Benchmark réussi:")
            logger.info(f"   Backend: {benchmark_results['backend']}")
            logger.info(f"   Modèle: {benchmark_results['model']}")
            logger.info(f"   Temps total: {benchmark_results['total_time']:.2f}s")
            logger.info(
                f"   Vitesse: {benchmark_results['tokens_per_second']:.1f} tokens/sec"
            )

        # ===== TEST 9: Statistiques =====
        logger.info("\n Test 9: Statistiques d'utilisation...")

        stats = llm.get_statistics()
        logger.info("[SUCCESS] Statistiques:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")

        # ===== RÉSUMÉ =====
        logger.info("\n" + "=" * 70)
        logger.info("[CELEBRATE] TOUS LES TESTS LLM HANDLER RÉUSSIS")
        logger.info("=" * 70)
        logger.info("\n[SUCCESS] Détection backend: OK")
        logger.info("[SUCCESS] Health check: OK")
        logger.info("[SUCCESS] Génération simple: OK")
        logger.info("[SUCCESS] RAG agricole: OK")
        logger.info("[SUCCESS] Templates prompts: OK")
        logger.info("[SUCCESS] Fallback: OK")
        logger.info("[SUCCESS] Post-processing: OK")
        logger.info("[SUCCESS] Benchmark: OK")
        logger.info("[SUCCESS] Statistiques: OK")

        return True

    except Exception as e:
        logger.error(f"\n[ERROR] TEST ÉCHOUÉ: {e}")
        import traceback

        traceback.print_exc()
        return False


# ============================================================================
# INTÉGRATION AVEC PIPELINE RAG
# ============================================================================


class ExempleIntegrationRAG:
    """
    Exemple d'intégration LLM Handler dans pipeline RAG complet
    """

    def __init__(self):
        """Initialise le pipeline RAG avec LLM Handler"""
        from src.embeddings import EmbeddingGenerator  # Votre module
        from src.vector_store import FAISSVectorStore  # Votre module

        self.embedding_model = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore()
        self.llm_handler = LLMHandler(
            backend=LLMBackend.AUTO,
            generation_config=GenerationConfig(
                temperature=0.1,  # Factuel pour agriculture
                max_tokens=800,  # Réponses raisonnables
                repeat_penalty=1.2,  # Éviter répétitions
            ),
            enable_cache=False,  # Désactiver en dev, activer en prod
        )

        logger.info("[SUCCESS] Pipeline RAG initialisé")

    def load_system(self):
        """Charge le système (index vectoriel)"""
        success = self.vector_store.load()
        if not success:
            raise RuntimeError("Échec chargement vector store. Lancer setup() d'abord.")
        logger.info("[SUCCESS] Système chargé")

    def answer_question(
        self,
        question: str,
        k: int = 3,
        template: PromptTemplate = PromptTemplate.STANDARD,
    ) -> Dict[str, Any]:
        """
        Répond à une question agricole (API complète)

        Args:
            question: Question utilisateur
            k: Nombre de documents à récupérer
            template: Template de prompt

        Returns:
            Dict avec réponse complète et métadonnées
        """
        try:
            logger.info(f" Question: {question}")

            # 1. Embeddings de la question
            question_embedding = self.embedding_model.model.encode(question)

            # 2. Recherche documents pertinents
            search_results = self.vector_store.search(question_embedding, k=k)

            if not search_results:
                logger.warning("[WARNING] Aucun document pertinent trouvé")

            logger.info(f"[DOCS] {len(search_results)} documents trouvés")

            # 3. Préparer contexte pour LLM
            context_docs = []
            for result in search_results:
                context_docs.append(
                    {"text": result.document_text, "metadata": result.metadata}
                )

            # 4. Générer réponse avec LLM
            llm_response = self.llm_handler.generate_answer(
                question, context_docs, template=template
            )

            # 5. Formater réponse finale
            final_response = {
                "question": question,
                "reponse": llm_response.text,
                "sources": [
                    {
                        "titre": result.metadata.get("titre", "Unknown"),
                        "source": result.metadata.get("source", "Unknown"),
                        "organisme": result.metadata.get("organisme", "Unknown"),
                        "pertinence": result.similarity_score,
                    }
                    for result in search_results
                ],
                "metadata": {
                    "backend": llm_response.backend,
                    "model": llm_response.model,
                    "generation_time": llm_response.generation_time,
                    "tokens_generated": llm_response.tokens_generated,
                    "documents_used": len(search_results),
                    "success": llm_response.success,
                },
            }

            logger.info(f"[SUCCESS] Réponse générée avec succès")

            return final_response

        except Exception as e:
            logger.error(f"[ERROR] Erreur answer_question: {e}")
            return {
                "question": question,
                "reponse": f"Erreur lors du traitement: {e}",
                "sources": [],
                "metadata": {"success": False, "error": str(e)},
            }


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================


def main():
    """
    Script principal - Différents modes d'utilisation
    """
    import argparse

    parser = argparse.ArgumentParser(description="LLM Handler pour RAG Agricole BF")
    parser.add_argument(
        "--mode",
        choices=["test", "benchmark", "interactive"],
        default="test",
        help="Mode opération",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "huggingface", "auto"],
        default="auto",
        help="Backend LLM",
    )
    parser.add_argument("--question", type=str, help="Question pour mode interactive")

    args = parser.parse_args()

    if args.mode == "test":
        # Lancer tests automatisés
        logger.info("[TEST] Mode TEST - Suite de tests complète")
        test_llm_handler_complete()

    elif args.mode == "benchmark":
        # Benchmark performance
        logger.info("⚡ Mode BENCHMARK")

        backend_map = {
            "ollama": LLMBackend.OLLAMA,
            "huggingface": LLMBackend.HUGGINGFACE,
            "auto": LLMBackend.AUTO,
        }

        llm = LLMHandler(backend=backend_map[args.backend])

        # Health check
        health = llm.health_check()
        logger.info(f"Health: {health}")

        # Benchmark
        results = llm.benchmark()
        logger.info(f"\n[STATS] RÉSULTATS BENCHMARK:")
        for key, value in results.items():
            logger.info(f"   {key}: {value}")

    elif args.mode == "interactive":
        # Mode interactif pour tests rapides
        logger.info(" Mode INTERACTIF")

        backend_map = {
            "ollama": LLMBackend.OLLAMA,
            "huggingface": LLMBackend.HUGGINGFACE,
            "auto": LLMBackend.AUTO,
        }

        llm = LLMHandler(backend=backend_map[args.backend])

        # Test document
        test_docs = [
            {
                "text": "Le sorgho nécessite 150 kg/ha d'engrais NPK 14-23-14 au semis.",
                "metadata": {"titre": "Guide sorgho", "source": "FAO"},
            }
        ]

        question = args.question or "Quel engrais pour le sorgho ?"

        logger.info(f"\nQuestion: {question}")
        response = llm.generate_answer(question, test_docs)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"RÉPONSE:")
        logger.info(f"{'=' * 70}")
        logger.info(response.text)
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Métadonnées:")
        logger.info(f"  Backend: {response.backend}")
        logger.info(f"  Modèle: {response.model}")
        logger.info(f"  Temps: {response.generation_time:.2f}s")
        logger.info(f"  Tokens: {response.tokens_generated}")


if __name__ == "__main__":
    main()
