import logging
import sys
import io
import requests
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

# Forcer UTF-8 console
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

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Types et Enums
class LLMBackend(Enum):
    OLLAMA = "ollama"

class PromptTemplate(Enum):
    STANDARD = "standard"
    CONCISE = "concise"
    DETAILED = "detailed"

@dataclass
class GenerationConfig:
    """
    Configuration génération LLM - Optimisée pour Llama3.2:3b
    """
    temperature: float = 0.3        # Légèrement plus haut pour plus de créativité
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 512           # Réduit pour éviter les timeouts
    num_ctx: int = 4096             # Contexte réduit pour stabilité
    repeat_penalty: float = 1.1     # Pénalité réduite
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = ["<|eot_id|>", "###", "Human:", "User:"]
    
    def to_ollama_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
            "repeat_penalty": self.repeat_penalty,
            "stop": self.stop_sequences
        }

@dataclass
class LLMResponse:
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
        return {
            'response': self.text,
            'model': self.model,
            'backend': self.backend,
            'generation_time': self.generation_time,
            'tokens_generated': self.tokens_generated,
            'tokens_per_second': self.tokens_per_second,
            'context_used': self.context_used,
            'sources': self.sources,
            'success': self.success,
            'error': self.error
        }

class OllamaHandler:
    """
    Handler LLM pour Ollama avec modèle Llama3.2:3b
    Version corrigée pour les erreurs 500
    """
    
    OLLAMA_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.2:3b"
    REQUEST_TIMEOUT = 60            # Timeout augmenté
    MAX_RETRIES = 2
    RETRY_DELAY = 3
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        generation_config: Optional[GenerationConfig] = None
    ):
        self.model = model
        self.generation_config = generation_config or GenerationConfig()
        
        # Session HTTP avec timeout plus long
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
        })
        
        # Statistiques
        self.stats = {
            'total_requests': 0,
            'errors': 0,
            'retries': 0
        }
        
        # Vérifier que le modèle est disponible
        self._verify_model()
        
        logger.info(f"[SUCCESS] Ollama Handler initialisé - Modèle: {self.model}")
    
    def _verify_model(self) -> None:
        """Vérifie que le modèle Llama3.2 est disponible"""
        try:
            response = requests.get(
                f"{self.OLLAMA_BASE_URL}/api/tags",
                timeout=10
            )
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model not in model_names:
                    logger.warning(f"[WARNING] Modèle {self.model} non trouvé")
                    logger.warning(f"Modèles disponibles: {model_names}")
                    logger.info(f"💡 Pour installer: ollama pull {self.model}")
                    
                    # Suggérer des alternatives
                    alternatives = ["llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "llama3.2"]
                    for alt in alternatives:
                        if alt in model_names:
                            logger.info(f"🔄 Utilisation alternative: {alt}")
                            self.model = alt
                            break
                    else:
                        # Si aucun modèle alternatif, utiliser le premier disponible
                        if model_names:
                            logger.info(f"🔄 Utilisation du premier modèle disponible: {model_names[0]}")
                            self.model = model_names[0]
                else:
                    logger.info(f"✅ Modèle {self.model} trouvé et disponible")
            else:
                logger.warning("⚠️ Impossible de vérifier les modèles Ollama")
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification modèle: {e}")
            logger.info("💡 Vérifiez que Ollama est démarré: ollama serve")
    
    def _build_simple_prompt(
        self,
        question: str,
        context_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Construit un prompt SIMPLE et ROBUSTE pour Llama3.2
        Évite les formats complexes qui peuvent causer des erreurs 500
        """
        # Construire contexte de manière simple
        context_parts = []
        for i, doc in enumerate(context_docs[:3]):  # Limiter à 3 documents
            text = doc.get('text', doc.get('contenu', ''))
            # Limiter la longueur du texte
            text_excerpt = text[:500] if len(text) > 500 else text
            context_parts.append(f"Document {i+1}: {text_excerpt}")
        
        context_text = "\n\n".join(context_parts)
        
        # Prompt SIMPLE sans formatage complexe
        prompt = f"""Contexte technique sur l'agriculture au Burkina Faso:

{context_text}

Question: {question}

En tant qu'expert agricole pour le Burkina Faso, réponds en français de façon claire et pratique en te basant sur les documents ci-dessus. Sois précis et donne des conseils applicables.

Réponse:"""
        
        return prompt
    
    def _build_agricultural_prompt(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        template: PromptTemplate = PromptTemplate.STANDARD
    ) -> str:
        """
        Construit un prompt optimisé pour Llama3.2 et l'agriculture BF
        Version simplifiée pour éviter les erreurs 500
        """
        # Construire section contexte
        context_parts = []
        
        for i, doc in enumerate(context_docs[:3]):  # Réduit à 3 documents max
            text = doc.get('text', doc.get('contenu', ''))
            metadata = doc.get('metadata', {})
            
            source = metadata.get('titre', f'Document {i+1}')
            # Limiter la longueur du texte
            text_excerpt = text[:400] if len(text) > 400 else text
            
            context_parts.append(f"[{source}]\n{text_excerpt}")
        
        context_text = "\n\n".join(context_parts)
        
        # Templates SIMPLIFIÉS pour éviter les erreurs
        if template == PromptTemplate.CONCISE:
            prompt = f"""Contexte:
{context_text}

Question: {question}

Réponds de façon concise (3-5 phrases) en tant que conseiller agricole pour le Burkina Faso:"""
        
        elif template == PromptTemplate.DETAILED:
            prompt = f"""Documents de référence sur l'agriculture burkinabè:
{context_text}

Question: {question}

En tant qu'expert agricole, fournis une réponse détaillée et structurée:
1. Réponse principale
2. Explications techniques  
3. Conseils pratiques
4. Sources utilisées

Réponse:"""
        
        else:  # STANDARD
            prompt = f"""Base de connaissances agricoles Burkina Faso:
{context_text}

Question: {question}

En tant que conseiller agricole expert, réponds en français de façon claire et pratique. Base ta réponse sur les documents ci-dessus. Sois précis sur les techniques, quantités et périodes.

Réponse:"""
        
        return prompt
    
    def generate_answer(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        template: PromptTemplate = PromptTemplate.STANDARD,
        use_simple_prompt: bool = True  # Option pour utiliser le prompt simple
    ) -> LLMResponse:
        """
        Génère une réponse avec Ollama Llama3.2
        Version robuste avec gestion d'erreurs améliorée
        """
        try:
            self.stats['total_requests'] += 1
            
            if not question or not question.strip():
                raise ValueError("Question vide")
            
            if not context_docs:
                return self._generate_fallback_response(question)
            
            # Construire prompt (simple par défaut pour plus de stabilité)
            if use_simple_prompt:
                prompt = self._build_simple_prompt(question, context_docs)
            else:
                prompt = self._build_agricultural_prompt(question, context_docs, template)
            
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Générer avec Ollama
            raw_response, metadata = self._generate_with_ollama(prompt)
            
            # Post-processing
            cleaned_response, sources = self._post_process_response(raw_response, context_docs)
            
            # Construire objet réponse
            llm_response = LLMResponse(
                text=cleaned_response,
                model=metadata.get('model', self.model),
                backend='ollama',
                generation_time=metadata.get('generation_time', 0),
                tokens_generated=metadata.get('tokens_generated', 0),
                tokens_per_second=metadata.get('tokens_per_second', 0),
                context_used=True,
                sources=sources,
                success=True
            )
            
            logger.info(f"✅ Réponse générée: '{question[:40]}...'")
            
            return llm_response
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Échec génération: {e}")
            return self._generate_error_response(question, str(e))
    
    def _generate_with_ollama(
        self,
        prompt: str,
        retry_count: int = 0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Génère réponse avec Ollama et Llama3.2
        Version avec meilleure gestion d'erreurs
        """
        try:
            # Payload SIMPLIFIÉ pour éviter les erreurs
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.generation_config.temperature,
                    "top_p": self.generation_config.top_p,
                    "num_predict": self.generation_config.max_tokens,
                }
            }
            
            logger.info(f"🔄 Génération avec {self.model} (tentative {retry_count + 1})...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=self.REQUEST_TIMEOUT
            )
            
            # Vérifier le statut HTTP
            if response.status_code != 200:
                error_msg = f"Erreur HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('error', 'No details')
                    error_msg += f": {error_detail}"
                except:
                    error_msg += f": {response.text}"
                
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            data = response.json()
            
            generation_time = time.time() - start_time
            
            # Métadonnées
            metadata = {
                'model': data.get('model', self.model),
                'generation_time': generation_time,
                'total_duration': data.get('total_duration', 0) / 1e9,
                'tokens_generated': data.get('eval_count', 0),
                'tokens_prompt': data.get('prompt_eval_count', 0),
                'tokens_per_second': data.get('eval_count', 0) / generation_time if generation_time > 0 else 0
            }
            
            logger.info(f"✅ Génération réussie: {metadata['tokens_generated']} tokens en {generation_time:.2f}s")
            
            return data.get('response', ''), metadata
            
        except requests.exceptions.Timeout:
            logger.warning(f"⏱️ Timeout Ollama (tentative {retry_count + 1})")
            if retry_count < self.MAX_RETRIES:
                self.stats['retries'] += 1
                time.sleep(self.RETRY_DELAY)
                return self._generate_with_ollama(prompt, retry_count + 1)
            else:
                raise TimeoutError("Ollama timeout après plusieurs tentatives")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"🔌 Erreur connexion Ollama")
            if retry_count < self.MAX_RETRIES:
                self.stats['retries'] += 1
                time.sleep(self.RETRY_DELAY * 2)  # Attente plus longue pour connexion
                return self._generate_with_ollama(prompt, retry_count + 1)
            else:
                raise ConnectionError("Impossible de se connecter à Ollama")
                
        except Exception as e:
            logger.error(f"❌ Erreur Ollama: {e}")
            if retry_count < self.MAX_RETRIES:
                self.stats['retries'] += 1
                time.sleep(self.RETRY_DELAY)
                return self._generate_with_ollama(prompt, retry_count + 1)
            else:
                raise
    
    def _post_process_response(
        self,
        raw_response: str,
        context_docs: List[Dict]
    ) -> Tuple[str, List[str]]:
        """
        Post-traite la réponse brute de Llama3.2
        """
        response = raw_response.strip()
        
        # Nettoyer la réponse
        for stop_seq in ['###', '---', 'Question:', 'User:', 'Human:']:
            if stop_seq in response:
                response = response.split(stop_seq)[0].strip()
        
        # Limiter longueur excessive
        if len(response) > 2000:
            response = response[:2000] + "..."
        
        # Extraire sources mentionnées
        sources = []
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            source_name = metadata.get('titre', '') or metadata.get('source', '')
            if source_name and source_name.lower() in response.lower():
                sources.append(source_name)
        
        # Si aucune source, utiliser les sources du contexte
        if not sources:
            sources = [
                doc.get('metadata', {}).get('titre', 'Document') 
                for doc in context_docs[:2]
            ]
        
        # Nettoyer espaces
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)
        response = response.strip()
        
        return response, sources
    
    def _generate_fallback_response(self, question: str) -> LLMResponse:
        """Génère réponse fallback quand pas de contexte"""
        fallback_text = (
            "Je n'ai pas trouvé d'informations spécifiques dans ma base de connaissances "
            "agricoles pour répondre à votre question sur le Burkina Faso. "
            "Pour des conseils précis, je vous recommande de consulter les services "
            "agricoles locaux ou les organisations spécialisées."
        )
        
        return LLMResponse(
            text=fallback_text,
            model=self.model,
            backend='fallback',
            generation_time=0,
            tokens_generated=len(fallback_text.split()),
            tokens_per_second=0,
            context_used=False,
            sources=[],
            success=True
        )
    
    def _generate_error_response(self, question: str, error: str) -> LLMResponse:
        """Génère réponse d'erreur"""
        error_text = (
            "Désolé, je rencontre actuellement des difficultés techniques. "
            "Veuillez réessayer dans quelques instants."
        )
        
        return LLMResponse(
            text=error_text,
            model='error',
            backend='error',
            generation_time=0,
            tokens_generated=0,
            tokens_per_second=0,
            context_used=False,
            sources=[],
            success=False,
            error=error
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Vérifie l'état d'Ollama et du modèle"""
        try:
            response = requests.get(
                f"{self.OLLAMA_BASE_URL}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_available = self.model in [m['name'] for m in models]
                
                return {
                    'status': 'healthy' if model_available else 'model_missing',
                    'model_available': model_available,
                    'available_models': [m['name'] for m in models],
                    'current_model': self.model
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return {'status': 'unavailable'}
    
    def test_simple_generation(self) -> bool:
        """
        Test simple de génération avec une requête basique
        """
        try:
            test_prompt = "Explique l'agriculture en une phrase."
            
            payload = {
                "model": self.model,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50
                }
            }
            
            response = self.session.post(
                f"{self.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Test simple réussi: {data.get('response', '')[:50]}...")
                return True
            else:
                logger.error(f"❌ Test simple échoué: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test simple échoué: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne statistiques d'utilisation"""
        error_rate = (
            self.stats['errors'] / self.stats['total_requests'] * 100 
            if self.stats['total_requests'] > 0 else 0
        )
        
        return {
            **self.stats,
            'error_rate': f"{error_rate:.1f}%",
            'current_model': self.model
        }

def test_llama_handler():
    """Test robuste du handler Llama3.2"""
    logger.info("🧪 Test Ollama Handler avec Llama3.2...")
    
    try:
        handler = OllamaHandler(model="llama3.2:3b")
        
        # Health check détaillé
        health = handler.health_check()
        logger.info(f"État Ollama: {health['status']}")
        logger.info(f"Modèle courant: {health['current_model']}")
        
        if health['status'] != 'healthy':
            logger.error("❌ Ollama ou le modèle n'est pas disponible")
            return False
        
        # Test simple de génération d'abord
        logger.info("🔧 Test simple de génération...")
        simple_test = handler.test_simple_generation()
        if not simple_test:
            logger.error("❌ Test simple échoué - vérifiez Ollama")
            return False
        
        # Test complet avec contexte
        logger.info("🧪 Test complet avec contexte...")
        test_docs = [{
            'text': "Le mil est une céréale résistante à la sécheresse. Au Burkina Faso, il se sème en juin-juillet avec 100-150 kg/ha d'engrais NPK.",
            'metadata': {'titre': 'Culture du mil', 'source': 'Guide FAO'}
        }]
        
        question = "Quand semer le mil au Burkina Faso ?"
        response = handler.generate_answer(
            question, 
            test_docs,
            use_simple_prompt=True  # Utiliser le prompt simple pour le test
        )
        
        if response.success:
            logger.info("✅ Test complet réussi avec Llama3.2")
            logger.info(f"Question: {question}")
            logger.info(f"Réponse: {response.text}")
            logger.info(f"Performance: {response.tokens_per_second:.1f} tokens/sec")
            
            stats = handler.get_statistics()
            logger.info(f"Statistiques: {stats}")
            
            return True
        else:
            logger.error(f"❌ Échec génération: {response.error}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur test: {e}")
        return False

def main():
    """Script principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ollama Handler pour Llama3.2')
    parser.add_argument('--mode', choices=['test', 'interactive', 'health', 'simple-test'],
                       default='test', help='Mode opération')
    parser.add_argument('--question', type=str,
                       help='Question pour mode interactive')
    parser.add_argument('--model', type=str, default="llama3.2:3b",
                       help='Modèle Ollama à utiliser')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        success = test_llama_handler()
        sys.exit(0 if success else 1)
        
    elif args.mode == 'health':
        handler = OllamaHandler(model=args.model)
        health = handler.health_check()
        print(f"\n🩺 HEALTH CHECK:")
        print(f"Statut: {health['status']}")
        print(f"Modèle courant: {health['current_model']}")
        print(f"Modèle disponible: {health['model_available']}")
        print(f"Modèles installés: {health.get('available_models', [])}")
        
    elif args.mode == 'simple-test':
        handler = OllamaHandler(model=args.model)
        success = handler.test_simple_generation()
        print(f"Test simple: {'✅ RÉUSSI' if success else '❌ ÉCHEC'}")
        
    elif args.mode == 'interactive':
        handler = OllamaHandler(model=args.model)
        
        question = args.question or "Quel engrais pour le sorgho ?"
        test_docs = [{
            'text': "Le sorgho nécessite 150 kg/ha d'engrais NPK 14-23-14 au semis.",
            'metadata': {'titre': 'Fertilisation sorgho', 'source': 'CIRAD'}
        }]
        
        response = handler.generate_answer(question, test_docs, use_simple_prompt=True)
        
        print(f"\n❓ QUESTION: {question}")
        print(f"\n🤖 RÉPONSE (Llama3.2):")
        print("=" * 70)
        print(response.text)
        print("=" * 70)
        print(f"\n📈 MÉTRIQUES:")
        print(f"  Modèle: {response.model}")
        print(f"  Temps: {response.generation_time:.2f}s")
        print(f"  Tokens: {response.tokens_generated}")
        print(f"  Vitesse: {response.tokens_per_second:.1f} tokens/sec")
        print(f"  Succès: {response.success}")

if __name__ == "__main__":
    main()