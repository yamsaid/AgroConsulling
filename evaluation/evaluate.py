"""
Système d'évaluation pour RAG Agricole Burkina Faso
Évalue la performance du système de questions-réponses

Métriques évaluées:
- Temps de réponse (latence)
- Précision du retrieval (pertinence des documents)
- Qualité de la génération LLM
- Taux de succès
- Couverture des sources

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --quick  # Évaluation rapide
    python evaluation/evaluate.py --full   # Évaluation complète
"""

import json
import time
import csv
import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import statistics
import requests

# Configuration
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET DE TEST
# ============================================================================

TEST_QUESTIONS = [
    # Questions agricoles basiques
    {
        "id": 1,
        "question": "Quel engrais utiliser pour le mil?",
        "category": "engrais",
        "difficulty": "facile",
        "expected_keywords": ["NPK", "engrais", "mil", "dose"],
    },
    {
        "id": 2,
        "question": "Comment lutter contre les ravageurs du maïs?",
        "category": "ravageurs",
        "difficulty": "moyen",
        "expected_keywords": ["ravageur", "maïs", "traitement", "protection"],
    },
    {
        "id": 3,
        "question": "Quelles sont les bonnes pratiques pour la culture du riz?",
        "category": "culture",
        "difficulty": "moyen",
        "expected_keywords": ["riz", "culture", "pratique", "semis"],
    },
    {
        "id": 4,
        "question": "Quelle est la dose d'urée recommandée pour le sorgho?",
        "category": "engrais",
        "difficulty": "difficile",
        "expected_keywords": ["urée", "dose", "sorgho", "kg"],
    },
    {
        "id": 5,
        "question": "Comment préparer le sol pour la culture de l'arachide?",
        "category": "culture",
        "difficulty": "moyen",
        "expected_keywords": ["sol", "arachide", "préparation", "labour"],
    },
    
    # Questions de précision
    {
        "id": 6,
        "question": "Quand semer le coton au Burkina Faso?",
        "category": "culture",
        "difficulty": "moyen",
        "expected_keywords": ["coton", "semis", "période", "saison"],
    },
    {
        "id": 7,
        "question": "Quels sont les symptômes de la rouille du maïs?",
        "category": "maladie",
        "difficulty": "difficile",
        "expected_keywords": ["rouille", "maïs", "symptôme", "feuille"],
    },
    {
        "id": 8,
        "question": "Comment améliorer la fertilité du sol?",
        "category": "sol",
        "difficulty": "moyen",
        "expected_keywords": ["fertilité", "sol", "compost", "fumier"],
    },
    
    # Questions hors domaine (pour tester le filtrage)
    {
        "id": 9,
        "question": "Bonjour, comment allez-vous?",
        "category": "salutation",
        "difficulty": "facile",
        "expected_keywords": ["bonjour", "assistant", "aide"],
        "is_out_of_domain": True,
    },
    {
        "id": 10,
        "question": "Qui es-tu?",
        "category": "auto-description",
        "difficulty": "facile",
        "expected_keywords": ["AgroConsulting", "assistant", "agriculture"],
        "is_out_of_domain": True,
    },
]


# ============================================================================
# DATACLASSES POUR LES RÉSULTATS
# ============================================================================

@dataclass
class RetrievalMetrics:
    """Métriques de qualité du retrieval"""
    num_documents: int
    avg_similarity: float
    max_similarity: float
    min_similarity: float
    retrieval_time: float


@dataclass
class GenerationMetrics:
    """Métriques de qualité de la génération"""
    response_length: int
    generation_time: float
    tokens_generated: int
    tokens_per_second: float
    backend_used: str
    model_used: str


@dataclass
class QuestionResult:
    """Résultat pour une question"""
    question_id: int
    question: str
    category: str
    difficulty: str
    
    # Succès
    success: bool
    error_message: Optional[str]
    
    # Temps
    total_time: float
    retrieval_time: float
    generation_time: float
    
    # Retrieval
    num_documents_retrieved: int
    avg_similarity_score: float
    
    # Génération
    response_text: str
    response_length: int
    tokens_generated: int
    tokens_per_second: float
    
    # Qualité
    has_sources: bool
    num_sources: int
    keyword_coverage: float  # % de mots-clés attendus trouvés
    
    # Métadonnées
    backend_used: str
    model_used: str
    timestamp: str


@dataclass
class EvaluationSummary:
    """Résumé de l'évaluation"""
    total_questions: int
    successful_questions: int
    failed_questions: int
    success_rate: float
    
    # Temps
    avg_total_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    
    # Retrieval
    avg_documents_retrieved: float
    avg_similarity_score: float
    
    # Génération
    avg_response_length: float
    avg_tokens_generated: float
    avg_tokens_per_second: float
    
    # Qualité
    questions_with_sources: int
    avg_keyword_coverage: float
    
    # Par catégorie
    results_by_category: Dict[str, Dict[str, Any]]
    results_by_difficulty: Dict[str, Dict[str, Any]]
    
    # Timestamp
    evaluation_date: str
    duration: float


# ============================================================================
# CLASSE D'ÉVALUATION
# ============================================================================

class RAGEvaluator:
    """Évaluateur du système RAG"""
    
    def __init__(self, api_url: str = "http://localhost:8000", timeout: int = 60):
        self.api_url = api_url
        self.timeout = timeout
        self.results: List[QuestionResult] = []
        
    def check_api_health(self) -> bool:
        """Vérifie que l'API est en ligne"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ API Health: {health_data['status']}")
                
                # Afficher infos composants
                components = health_data.get("components", {})
                for name, info in components.items():
                    status = info.get("status", "unknown")
                    logger.info(f"   - {name}: {status}")
                    
                    if name == "vector_store":
                        docs = info.get("documents", 0)
                        logger.info(f"     Documents: {docs}")
                
                return health_data["status"] in ["healthy", "degraded"]
            else:
                logger.error(f"❌ API Health Check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Impossible de contacter l'API: {e}")
            return False
    
    def evaluate_question(self, question_data: Dict) -> QuestionResult:
        """Évalue une question"""
        question_id = question_data["id"]
        question = question_data["question"]
        category = question_data["category"]
        difficulty = question_data["difficulty"]
        expected_keywords = question_data.get("expected_keywords", [])
        is_ood = question_data.get("is_out_of_domain", False)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Question {question_id}: {question}")
        logger.info(f"Catégorie: {category} | Difficulté: {difficulty}")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            # Appeler l'API
            response = requests.post(
                f"{self.api_url}/ask",
                json={"question": question, "max_results": 3, "verbose": False},
                timeout=self.timeout
            )
            
            total_time = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"❌ Erreur HTTP {response.status_code}")
                return self._create_error_result(
                    question_data, f"HTTP {response.status_code}", total_time
                )
            
            data = response.json()
            
            # Extraire métriques
            success = data.get("success", False)
            response_text = data.get("reponse", "")
            sources = data.get("sources", [])
            metadata = data.get("metadata", {})
            
            # Calcul keyword coverage
            keyword_coverage = self._calculate_keyword_coverage(
                response_text, expected_keywords
            )
            
            # Créer résultat
            result = QuestionResult(
                question_id=question_id,
                question=question,
                category=category,
                difficulty=difficulty,
                success=success,
                error_message=None if success else "Échec de génération",
                total_time=total_time,
                retrieval_time=metadata.get("processing_time", 0) - metadata.get("generation_time", 0),
                generation_time=metadata.get("generation_time", 0),
                num_documents_retrieved=metadata.get("documents_used", 0),
                avg_similarity_score=0.0,  # Pas disponible directement
                response_text=response_text[:200] + "..." if len(response_text) > 200 else response_text,
                response_length=len(response_text),
                tokens_generated=metadata.get("tokens_generated", 0),
                tokens_per_second=metadata.get("tokens_per_second", 0),
                has_sources=len(sources) > 0,
                num_sources=len(sources),
                keyword_coverage=keyword_coverage,
                backend_used=metadata.get("backend", "unknown"),
                model_used=metadata.get("model", "unknown"),
                timestamp=datetime.now().isoformat(),
            )
            
            # Afficher résultat
            self._log_result(result)
            
            return result
            
        except requests.Timeout:
            total_time = time.time() - start_time
            logger.error(f"❌ Timeout après {total_time:.2f}s")
            return self._create_error_result(question_data, "Timeout", total_time)
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Erreur: {e}")
            return self._create_error_result(question_data, str(e), total_time)
    
    def _calculate_keyword_coverage(self, text: str, keywords: List[str]) -> float:
        """Calcule le % de mots-clés attendus présents dans le texte"""
        if not keywords:
            return 1.0
        
        text_lower = text.lower()
        found = sum(1 for kw in keywords if kw.lower() in text_lower)
        return found / len(keywords)
    
    def _create_error_result(self, question_data: Dict, error_msg: str, total_time: float) -> QuestionResult:
        """Crée un résultat d'erreur"""
        return QuestionResult(
            question_id=question_data["id"],
            question=question_data["question"],
            category=question_data["category"],
            difficulty=question_data["difficulty"],
            success=False,
            error_message=error_msg,
            total_time=total_time,
            retrieval_time=0,
            generation_time=0,
            num_documents_retrieved=0,
            avg_similarity_score=0,
            response_text="",
            response_length=0,
            tokens_generated=0,
            tokens_per_second=0,
            has_sources=False,
            num_sources=0,
            keyword_coverage=0,
            backend_used="N/A",
            model_used="N/A",
            timestamp=datetime.now().isoformat(),
        )
    
    def _log_result(self, result: QuestionResult):
        """Affiche les résultats dans les logs"""
        status = "✅ SUCCÈS" if result.success else "❌ ÉCHEC"
        logger.info(f"\n{status}")
        logger.info(f"Temps total: {result.total_time:.3f}s")
        logger.info(f"  - Retrieval: {result.retrieval_time:.3f}s")
        logger.info(f"  - Génération: {result.generation_time:.3f}s")
        logger.info(f"Documents: {result.num_documents_retrieved}")
        logger.info(f"Sources: {result.num_sources}")
        logger.info(f"Tokens: {result.tokens_generated} ({result.tokens_per_second:.1f} tok/s)")
        logger.info(f"Couverture mots-clés: {result.keyword_coverage:.1%}")
        logger.info(f"Backend: {result.backend_used} | Model: {result.model_used}")
        if not result.success:
            logger.error(f"Erreur: {result.error_message}")
    
    def evaluate_system(self, questions: Optional[List[Dict]] = None) -> EvaluationSummary:
        """Évalue le système complet"""
        
        if questions is None:
            questions = TEST_QUESTIONS
        
        logger.info("=" * 70)
        logger.info("DÉMARRAGE ÉVALUATION DU SYSTÈME RAG")
        logger.info("=" * 70)
        logger.info(f"Nombre de questions: {len(questions)}")
        logger.info(f"API URL: {self.api_url}")
        
        # Vérifier santé API
        if not self.check_api_health():
            logger.error("❌ API non disponible - Arrêt de l'évaluation")
            raise RuntimeError("API non disponible")
        
        start_time = time.time()
        self.results = []
        
        # Évaluer chaque question
        for i, question_data in enumerate(questions, 1):
            logger.info(f"\n[{i}/{len(questions)}] Évaluation en cours...")
            result = self.evaluate_question(question_data)
            self.results.append(result)
            time.sleep(0.5)  # Petit délai entre questions
        
        duration = time.time() - start_time
        
        # Calculer statistiques
        summary = self._compute_summary(duration)
        
        # Afficher résumé
        self._log_summary(summary)
        
        return summary
    
    def _compute_summary(self, duration: float) -> EvaluationSummary:
        """Calcule les statistiques globales"""
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        # Statistiques globales
        success_rate = len(successful) / len(self.results) if self.results else 0
        
        # Temps
        avg_total_time = statistics.mean([r.total_time for r in self.results]) if self.results else 0
        avg_retrieval_time = statistics.mean([r.retrieval_time for r in successful]) if successful else 0
        avg_generation_time = statistics.mean([r.generation_time for r in successful]) if successful else 0
        
        # Retrieval
        avg_documents = statistics.mean([r.num_documents_retrieved for r in successful]) if successful else 0
        avg_similarity = statistics.mean([r.avg_similarity_score for r in successful]) if successful else 0
        
        # Génération
        avg_response_length = statistics.mean([r.response_length for r in successful]) if successful else 0
        avg_tokens = statistics.mean([r.tokens_generated for r in successful]) if successful else 0
        avg_tok_per_sec = statistics.mean([r.tokens_per_second for r in successful]) if successful else 0
        
        # Qualité
        with_sources = sum(1 for r in successful if r.has_sources)
        avg_keyword_cov = statistics.mean([r.keyword_coverage for r in successful]) if successful else 0
        
        # Par catégorie
        categories = {}
        for result in self.results:
            cat = result.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        results_by_category = {}
        for cat, results in categories.items():
            succ = [r for r in results if r.success]
            results_by_category[cat] = {
                "total": len(results),
                "success": len(succ),
                "success_rate": len(succ) / len(results) if results else 0,
                "avg_time": statistics.mean([r.total_time for r in results]),
            }
        
        # Par difficulté
        difficulties = {}
        for result in self.results:
            diff = result.difficulty
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(result)
        
        results_by_difficulty = {}
        for diff, results in difficulties.items():
            succ = [r for r in results if r.success]
            results_by_difficulty[diff] = {
                "total": len(results),
                "success": len(succ),
                "success_rate": len(succ) / len(results) if results else 0,
                "avg_time": statistics.mean([r.total_time for r in results]),
            }
        
        return EvaluationSummary(
            total_questions=len(self.results),
            successful_questions=len(successful),
            failed_questions=len(failed),
            success_rate=success_rate,
            avg_total_time=avg_total_time,
            avg_retrieval_time=avg_retrieval_time,
            avg_generation_time=avg_generation_time,
            avg_documents_retrieved=avg_documents,
            avg_similarity_score=avg_similarity,
            avg_response_length=avg_response_length,
            avg_tokens_generated=avg_tokens,
            avg_tokens_per_second=avg_tok_per_sec,
            questions_with_sources=with_sources,
            avg_keyword_coverage=avg_keyword_cov,
            results_by_category=results_by_category,
            results_by_difficulty=results_by_difficulty,
            evaluation_date=datetime.now().isoformat(),
            duration=duration,
        )
    
    def _log_summary(self, summary: EvaluationSummary):
        """Affiche le résumé"""
        logger.info("\n" + "=" * 70)
        logger.info("RÉSUMÉ DE L'ÉVALUATION")
        logger.info("=" * 70)
        logger.info(f"📊 Questions totales: {summary.total_questions}")
        logger.info(f"✅ Succès: {summary.successful_questions} ({summary.success_rate:.1%})")
        logger.info(f"❌ Échecs: {summary.failed_questions}")
        logger.info(f"⏱️  Durée totale: {summary.duration:.2f}s")
        
        logger.info(f"\n⏱️  TEMPS MOYENS")
        logger.info(f"   Total: {summary.avg_total_time:.3f}s")
        logger.info(f"   Retrieval: {summary.avg_retrieval_time:.3f}s")
        logger.info(f"   Génération: {summary.avg_generation_time:.3f}s")
        
        logger.info(f"\n📚 RETRIEVAL")
        logger.info(f"   Documents moyens: {summary.avg_documents_retrieved:.1f}")
        logger.info(f"   Similarité moyenne: {summary.avg_similarity_score:.3f}")
        
        logger.info(f"\n💬 GÉNÉRATION")
        logger.info(f"   Longueur moyenne: {summary.avg_response_length:.0f} chars")
        logger.info(f"   Tokens moyens: {summary.avg_tokens_generated:.0f}")
        logger.info(f"   Vitesse: {summary.avg_tokens_per_second:.1f} tok/s")
        
        logger.info(f"\n✨ QUALITÉ")
        logger.info(f"   Questions avec sources: {summary.questions_with_sources}/{summary.successful_questions}")
        logger.info(f"   Couverture mots-clés: {summary.avg_keyword_coverage:.1%}")
        
        logger.info(f"\n📂 PAR CATÉGORIE")
        for cat, stats in summary.results_by_category.items():
            logger.info(f"   {cat}: {stats['success']}/{stats['total']} ({stats['success_rate']:.1%}) - {stats['avg_time']:.3f}s")
        
        logger.info(f"\n📊 PAR DIFFICULTÉ")
        for diff, stats in summary.results_by_difficulty.items():
            logger.info(f"   {diff}: {stats['success']}/{stats['total']} ({stats['success_rate']:.1%}) - {stats['avg_time']:.3f}s")
        
        logger.info("=" * 70)
    
    def export_results(self, output_dir: str = "evaluation_results"):
        """Exporte les résultats en JSON et CSV"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export JSON détaillé
        json_file = output_path / f"evaluation_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": [asdict(r) for r in self.results],
                    "summary": asdict(self._compute_summary(0))
                },
                f,
                indent=2,
                ensure_ascii=False
            )
        logger.info(f"✅ Résultats JSON exportés: {json_file}")
        
        # Export CSV simple
        csv_file = output_path / f"evaluation_{timestamp}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question_id", "question", "category", "difficulty",
                "success", "total_time", "num_documents", "num_sources",
                "tokens_generated", "tokens_per_second", "keyword_coverage"
            ])
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    "question_id": r.question_id,
                    "question": r.question,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "success": r.success,
                    "total_time": round(r.total_time, 3),
                    "num_documents": r.num_documents_retrieved,
                    "num_sources": r.num_sources,
                    "tokens_generated": r.tokens_generated,
                    "tokens_per_second": round(r.tokens_per_second, 1),
                    "keyword_coverage": round(r.keyword_coverage, 2),
                })
        logger.info(f"✅ Résultats CSV exportés: {csv_file}")
        
        return json_file, csv_file


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale d'évaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Évaluation du système RAG")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL de l'API")
    parser.add_argument("--quick", action="store_true", help="Évaluation rapide (5 questions)")
    parser.add_argument("--full", action="store_true", help="Évaluation complète (toutes questions)")
    parser.add_argument("--output-dir", default="evaluation_results", help="Dossier de sortie")
    
    args = parser.parse_args()
    
    # Sélectionner questions
    if args.quick:
        questions = TEST_QUESTIONS[:5]
        logger.info("Mode rapide: 5 questions")
    elif args.full:
        questions = TEST_QUESTIONS
        logger.info("Mode complet: toutes les questions")
    else:
        questions = TEST_QUESTIONS[:8]  # Par défaut
        logger.info("Mode par défaut: 8 questions")
    
    # Créer évaluateur
    evaluator = RAGEvaluator(api_url=args.api_url)
    
    try:
        # Lancer évaluation
        summary = evaluator.evaluate_system(questions)
        
        # Exporter résultats
        evaluator.export_results(output_dir=args.output_dir)
        
        logger.info("\n✅ ÉVALUATION TERMINÉE AVEC SUCCÈS!")
        
        # Code de sortie selon succès
        if summary.success_rate >= 0.8:
            logger.info("🎉 Excellent! Taux de succès >= 80%")
            return 0
        elif summary.success_rate >= 0.6:
            logger.warning("⚠️  Moyen. Taux de succès >= 60%")
            return 0
        else:
            logger.error("❌ Faible. Taux de succès < 60%")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'évaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())