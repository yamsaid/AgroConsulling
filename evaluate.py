import json
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import difflib

# ==================== CONFIGURATION ==================== #

TEST_DATA_PATH = Path("evaluation/test_questions.json")
RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== CHARGEMENT DONNÉES ==================== #


def load_test_questions() -> Dict:
    """Charge les questions de test depuis le JSON"""
    try:
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {TEST_DATA_PATH}")
        return {"questions": []}


# ==================== MÉTRIQUES D'ÉVALUATION ==================== #


def similarity_score(text1: str, text2: str) -> float:
    """
    Calcule la similarité entre deux textes (0-1)
    Utilise difflib pour une comparaison sémantique approximative
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalisation
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    # Utiliser SequenceMatcher de difflib
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def keyword_match_score(response: str, keywords: List[str]) -> float:
    """
    Évalue le pourcentage de mots-clés trouvés dans la réponse
    """
    if not keywords:
        return 0.0
    
    response_lower = response.lower()
    matched = sum(1 for kw in keywords if kw.lower() in response_lower)
    return matched / len(keywords)


def response_length_score(response: str) -> float:
    """
    Évalue si la longueur de la réponse est raisonnable (100-500 mots)
    """
    word_count = len(response.split())
    
    if word_count < 30:
        return 0.3  # Réponse trop courte
    elif word_count < 100:
        return 0.6
    elif word_count <= 500:
        return 1.0  # Optimal
    else:
        return 0.8  # Un peu long mais acceptable


def calculate_response_time_score(response_time: float) -> float:
    """
    Évalue le temps de réponse
    Optimal: < 2 secondes
    """
    if response_time < 1.0:
        return 1.0
    elif response_time < 2.0:
        return 0.9
    elif response_time < 5.0:
        return 0.7
    else:
        return 0.4


def evaluate_single_response(
    question: str,
    response: str,
    expected_answer: str,
    keywords: List[str],
    response_time: float = 1.0
) -> Dict:
    """
    Évalue une réponse unique avec plusieurs métriques
    """
    
    # Calcul des différentes métriques
    similarity = similarity_score(response, expected_answer)
    keyword_score = keyword_match_score(response, keywords)
    length_score = response_length_score(response)
    time_score = calculate_response_time_score(response_time)
    
    # Score composite (moyenne pondérée)
    composite_score = (
        similarity * 0.40 +
        keyword_score * 0.30 +
        length_score * 0.15 +
        time_score * 0.15
    )
    
    return {
        "similarity_score": round(similarity, 3),
        "keyword_score": round(keyword_score, 3),
        "length_score": round(length_score, 3),
        "time_score": round(time_score, 3),
        "composite_score": round(composite_score, 3),
        "response_time": round(response_time, 3),
    }


# ==================== FONCTION MOCK (SIMULATION) ==================== #


def mock_chat_response(question: str, test_data: Dict = None, perfect_mode: bool = False) -> Tuple[str, float]:
    """
    Simule une réponse du chatbot contextuelle basée sur la question
    Retourne la réponse et le temps écoulé
    
    Args:
        question: La question posée
        test_data: Données de test avec réponse attendue
        perfect_mode: Si True, utilise toujours les bonnes réponses
    """
    import random
    
    start_time = time.time()
    
    # Simulation du temps de traitement (0.5-2 secondes) - TOUJOURS exécuté
    delay = random.uniform(0.5, 2.0)
    time.sleep(delay)
    
    # Mode parfait: utilise toujours la réponse attendue
    if perfect_mode and test_data:
        response = test_data.get("expected_answer", "")
        elapsed_time = time.time() - start_time
        return response, elapsed_time
    
    # Mode normal: 70% bonnes réponses, 30% génériques
    if test_data:
        if random.random() > 0.3:  # 70% de chances d'avoir la bonne réponse
            response = test_data.get("expected_answer", "")
            elapsed_time = time.time() - start_time
            return response, elapsed_time
    
    # Réponses génériques contextuelles (fallback)
    mock_responses = [
        "Je vais analyser votre question sur l'agriculture. Cette question concerne les pratiques agricoles spécifiques. Les experts recommandent plusieurs approches basées sur les conditions locales et les ressources disponibles.",
        "Excellente question agricole! Pour répondre complètement, il faut considérer plusieurs facteurs: le climat local, le type de sol, les variétés disponibles, et les ressources de l'agriculteur. Les meilleures pratiques combinées donnent généralement les meilleurs résultats.",
        "Basé sur les connaissances agricoles, la réponse dépend de plusieurs paramètres. Les études montrent que l'approche adaptée au contexte local fonctionne mieux. Consultez les services agricoles locaux pour des recommandations spécifiques.",
    ]
    
    response = random.choice(mock_responses)
    elapsed_time = time.time() - start_time
    
    return response, elapsed_time


# ==================== FONCTION PRINCIPALE D'ÉVALUATION ==================== #


def evaluate_system(
    use_mock: bool = True,
    max_questions: int = None,
    perfect_mode: bool = False
) -> Dict:
    """
    Évalue le système complet avec les questions de test
    
    Args:
        use_mock: Utiliser les réponses simulées (True) ou appeler le backend (False)
        max_questions: Nombre max de questions à tester (None = toutes)
        perfect_mode: Si True, utilise les bonnes réponses (pour tester les métriques)
    
    Returns:
        Dict avec résultats d'évaluation détaillés
    """
    
    print("\n" + "="*60)
    print("🧪 ÉVALUATION DU SYSTÈME AgroConsolling")
    print("="*60)
    
    # Chargement des données
    data = load_test_questions()
    questions = data.get("questions", [])
    
    if not questions:
        print("❌ Aucune question de test disponible!")
        return {}
    
    # Limitation du nombre de questions
    if max_questions:
        questions = questions[:max_questions]
    
    print(f"\n📊 Configuration:")
    print(f"   • Nombre de questions: {len(questions)}")
    print(f"   • Mode: {'Simulation (mock)' if use_mock else 'Backend réel'}")
    print(f"   • Perfect Mode: {'✅ OUI (bonnes réponses)' if perfect_mode else '❌ NON (mixte)'}")
    print(f"\n" + "-"*60)
    
    # Évaluation
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "mode": "mock" if use_mock else "backend",
        },
        "evaluations": [],
        "statistics": {}
    }
    
    all_scores = {
        "similarity": [],
        "keyword": [],
        "length": [],
        "time": [],
        "composite": [],
    }
    
    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] Question: {q['question'][:50]}...")
        
        # Obtenir la réponse
        if use_mock:
            response, response_time = mock_chat_response(q["question"], test_data=q, perfect_mode=perfect_mode)
        else:
            # TODO: Appeler le vrai backend
            response, response_time = mock_chat_response(q["question"], test_data=q, perfect_mode=perfect_mode)
        
        # Évaluer la réponse
        eval_result = evaluate_single_response(
            question=q["question"],
            response=response,
            expected_answer=q["expected_answer"],
            keywords=q["keywords"],
            response_time=response_time
        )
        
        # Accumulation des scores
        all_scores["similarity"].append(eval_result["similarity_score"])
        all_scores["keyword"].append(eval_result["keyword_score"])
        all_scores["length"].append(eval_result["length_score"])
        all_scores["time"].append(eval_result["time_score"])
        all_scores["composite"].append(eval_result["composite_score"])
        
        # Statut
        status = "✅" if eval_result["composite_score"] >= 0.6 else "⚠️"
        print(f"   {status} Composite Score: {eval_result['composite_score']}")
        print(f"      • Similarité: {eval_result['similarity_score']}")
        print(f"      • Mots-clés: {eval_result['keyword_score']}")
        print(f"      • Longueur: {eval_result['length_score']}")
        print(f"      • Temps: {eval_result['time_score']} ({eval_result['response_time']}s)")
        
        results["evaluations"].append({
            "question_id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "response": response[:100] + "..." if len(response) > 100 else response,
            **eval_result
        })
    
    # Calcul des statistiques globales
    results["statistics"] = {
        "average_similarity": round(sum(all_scores["similarity"]) / len(all_scores["similarity"]), 3),
        "average_keyword": round(sum(all_scores["keyword"]) / len(all_scores["keyword"]), 3),
        "average_length": round(sum(all_scores["length"]) / len(all_scores["length"]), 3),
        "average_time": round(sum(all_scores["time"]) / len(all_scores["time"]), 3),
        "average_composite": round(sum(all_scores["composite"]) / len(all_scores["composite"]), 3),
        "success_rate": round(len([s for s in all_scores["composite"] if s >= 0.6]) / len(all_scores["composite"]), 3),
        "avg_response_time_seconds": round(sum([e["response_time"] for e in results["evaluations"]]) / len(results["evaluations"]), 3),
    }
    
    return results


# ==================== EXPORT RÉSULTATS ==================== #


def export_results_json(results: Dict, filename: str = None) -> Path:
    """Exporte les résultats en JSON"""
    if filename is None:
        filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    filepath = RESULTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Résultats JSON exportés: {filepath}")
    return filepath


def export_results_csv(results: Dict, filename: str = None) -> Path:
    """Exporte les résultats en CSV"""
    if filename is None:
        filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    filepath = RESULTS_DIR / filename
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "category",
                "question",
                "similarity_score",
                "keyword_score",
                "length_score",
                "time_score",
                "composite_score",
                "response_time"
            ]
        )
        writer.writeheader()
        
        for eval in results["evaluations"]:
            writer.writerow({
                "question_id": eval["question_id"],
                "category": eval["category"],
                "question": eval["question"],
                "similarity_score": eval["similarity_score"],
                "keyword_score": eval["keyword_score"],
                "length_score": eval["length_score"],
                "time_score": eval["time_score"],
                "composite_score": eval["composite_score"],
                "response_time": eval["response_time"]
            })
    
    print(f"✅ Résultats CSV exportés: {filepath}")
    return filepath


def print_summary(results: Dict):
    """Affiche un résumé des résultats"""
    stats = results.get("statistics", {})
    
    print("\n" + "="*60)
    print("📈 RÉSUMÉ DE L'ÉVALUATION")
    print("="*60)
    print(f"\n📊 Statistiques Globales:")
    print(f"   • Score Composite Moyen: {stats.get('average_composite', 0)}")
    print(f"   • Taux de Succès (≥0.6): {stats.get('success_rate', 0)*100:.1f}%")
    print(f"   • Similarité Moyenne: {stats.get('average_similarity', 0)}")
    print(f"   • Couverture Mots-clés: {stats.get('average_keyword', 0)*100:.1f}%")
    print(f"   • Temps Réponse Moyen: {stats.get('avg_response_time_seconds', 0)}s")
    print("\n" + "="*60 + "\n")


# ==================== POINT D'ENTRÉE ==================== #


if __name__ == "__main__":
    # Évaluation du système en MODE PARFAIT (pour tester les métriques)
    results = evaluate_system(use_mock=True, max_questions=20, perfect_mode=True)
    
    # Affichage du résumé
    print_summary(results)
    
    # Export des résultats
    if results:
        export_results_json(results)
        export_results_csv(results)
        
        print("\n✨ Évaluation terminée!")
        print(f"📁 Résultats disponibles dans: {RESULTS_DIR}")