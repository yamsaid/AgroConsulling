from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore
from src.llm_handler import LLMHandler, PromptTemplate, GenerationConfig

class RAGPipeline:
    def __init__(self):
        self.embedding_model = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore()
        
        # LLM Handler avec config optimisée agriculture
        self.llm = LLMHandler(
            generation_config=GenerationConfig(
                temperature=0.1,      # Factuel
                max_tokens=800,       # Réponses concises
                repeat_penalty=1.2    # Pas de répétitions
            )
        )
    
    def load(self):
        """Charge système existant"""
        self.vector_store.load()
        print("✅ Pipeline chargé")
    
    def answer(self, question: str, k: int = 3, verbose: bool = False) -> dict:
        """
        Réponse complète à une question agricole
        
        Args:
            question: Question utilisateur
            k: Nombre documents à récupérer
            verbose: Afficher détails
            
        Returns:
            dict avec réponse et métadonnées
        """
        # 1. Embeddings question
        if verbose:
            print(f"🔍 Embeddings question...")
        question_emb = self.embedding_model.model.encode(question)
        
        # 2. Recherche vectorielle
        if verbose:
            print(f"📚 Recherche {k} documents pertinents...")
        search_results = self.vector_store.search(question_emb, k=k)
        
        if not search_results:
            print("⚠️ Aucun document pertinent")
        
        # 3. Préparer contexte pour LLM
        context_docs = []
        for result in search_results:
            context_docs.append({
                'text': result.document_text,
                'metadata': result.metadata
            })
            
            if verbose:
                print(f"   - {result.metadata.get('titre', 'Unknown')} (score: {result.similarity_score:.3f})")
        
        # 4. Générer réponse LLM
        if verbose:
            print(f"🤖 Génération réponse LLM...")
        
        llm_response = self.llm.generate_answer(
            question,
            context_docs,
            template=PromptTemplate.STANDARD
        )
        
        # 5. Formater résultat final
        result = {
            'question': question,
            'reponse': llm_response.text,
            'sources': [
                {
                    'titre': r.metadata.get('titre', 'Unknown'),
                    'source': r.metadata.get('source', 'Unknown'),
                    'pertinence': r.similarity_score
                }
                for r in search_results
            ],
            'metadata': {
                'backend': llm_response.backend,
                'model': llm_response.model,
                'generation_time': llm_response.generation_time,
                'tokens_generated': llm_response.tokens_generated,
                'tokens_per_second': llm_response.tokens_per_second,
                'docs_used': len(search_results),
                'success': llm_response.success
            }
        }
        
        if verbose:
            print(f"✅ Réponse générée en {llm_response.generation_time:.2f}s")
            print(f"   Backend: {llm_response.backend}, Tokens: {llm_response.tokens_generated}")
        
        return result

# === UTILISATION ===
if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.load()
    
    # Test
    result = pipeline.answer(
        "Quel engrais utiliser pour le mil en saison sèche ?",
        k=3,
        verbose=True
    )
    
    print(f"\n{'='*70}")
    print(f"RÉPONSE:")
    print(f"{'='*70}")
    print(result['reponse'])
    print(f"\n{'='*70}")
    print(f"SOURCES:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['titre']} (pertinence: {source['pertinence']:.3f})")