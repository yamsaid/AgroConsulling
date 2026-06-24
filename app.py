"""
AgroConsulting - Hugging Face Spaces Deployment
Assistant IA Agricole pour le Burkina Faso

Point d'entrée principal pour Hugging Face Spaces
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Configuration pour Hugging Face Spaces
HF_SPACE = os.getenv("SPACE_ID") is not None
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")

# Chemins des données
DATA_DIR = Path("./data")
CORPUS_PATH = DATA_DIR / "corpus.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_INDEX_DIR = DATA_DIR / "faiss_db"

# Créer les répertoires si nécessaire
DATA_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)

# ============================================================================
# IMPORT DES MODULES
# ============================================================================

try:
    from src.embeddings import EmbeddingPipeline
    from src.vector_store import FAISSVectorStore
    from src.llm_handler import LLMHandler, LLMBackend, GenerationConfig
    from src.data_loader import ensure_data_available, load_data_from_hf
    logger.info("✅ Modules RAG importés avec succès")
except ImportError as e:
    logger.error(f"❌ Erreur import modules: {e}")
    raise

# Configuration données HF Hub
HF_DATA_REPO_ID = os.getenv("HF_DATA_REPO_ID") or os.getenv("HF_REPO_ID")

# ============================================================================
# INITIALISATION DU SYSTÈME RAG
# ============================================================================

class AgroConsultingRAG:
    """Classe principale pour le système RAG AgroConsulting"""
    
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.llm_handler = None
        self.initialized = False
        logger.info("🚀 Initialisation AgroConsulting RAG...")
    
    def initialize(self):
        """Initialise tous les composants du système RAG"""
        try:
            logger.info("=" * 70)
            logger.info("🔧 INITIALISATION DU SYSTÈME RAG")
            logger.info("=" * 70)
            
            # 0. Vérifier/charger les données
            if HF_DATA_REPO_ID:
                logger.info(f"📥 Vérification données depuis HF Hub: {HF_DATA_REPO_ID}")
                ensure_data_available(DATA_DIR, HF_DATA_REPO_ID)
            else:
                logger.info("📂 Utilisation données locales")
                if not CORPUS_PATH.exists():
                    logger.warning("⚠️ Corpus local non trouvé. Utilisez HF_DATA_REPO_ID pour charger depuis HF Hub")
            
            # 1. Modèle d'embeddings
            logger.info("📊 Chargement du modèle d'embeddings...")
            self.embedding_model = EmbeddingPipeline(str(CORPUS_PATH))
            self.embedding_model.initialize_embedding_model()
            logger.info(f"✅ Embeddings chargés: {self.embedding_model.model_name}")
            
            # 2. Vector store (FAISS)
            logger.info("🗂️ Chargement du vector store (FAISS)...")
            self.vector_store = FAISSVectorStore(str(FAISS_INDEX_DIR))
            
            # Vérifier si l'index existe
            index_exists = (
                (FAISS_INDEX_DIR / "faiss_index.index").exists() and
                (FAISS_INDEX_DIR / "corpus_data.pkl").exists()
            )
            
            if index_exists:
                logger.info("📂 Index FAISS existant détecté, chargement...")
                success = self.vector_store.load()
                if success:
                    stats = self.vector_store.get_statistics()
                    logger.info(f"✅ Vector store chargé: {stats.get('total_documents', 0)} documents")
                else:
                    logger.warning("⚠️ Échec chargement index, création nécessaire")
                    self._create_index()
            else:
                logger.info("📝 Index FAISS non trouvé, création...")
                self._create_index()
            
            # 3. LLM Handler (utiliser HuggingFace API sur HF Spaces)
            logger.info("🤖 Initialisation du LLM Handler...")
            
            # Utiliser un modèle plus léger pour HF Spaces (meilleure disponibilité)
            hf_model = os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
            
            self.llm_handler = LLMHandler(
                backend=LLMBackend.HUGGINGFACE,  # Utiliser HF API sur Spaces
                huggingface_model=hf_model,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    max_tokens=250,  # Réduit pour économiser les tokens
                    repeat_penalty=1.2,
                ),
                enable_cache=False,  # Désactiver cache sur Spaces (ressources limitées)
                hf_api_token=HF_TOKEN,
            )
            
            # Vérifier santé du LLM
            health = self.llm_handler.health_check()
            logger.info(f"✅ LLM Backend: {health['active_backend']}")
            
            self.initialized = True
            logger.info("=" * 70)
            logger.info("✅ SYSTÈME RAG INITIALISÉ AVEC SUCCÈS")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation: {e}")
            import traceback
            traceback.print_exc()
            self.initialized = False
            return False
    
    def _create_index(self):
        """Crée l'index FAISS depuis le corpus et les embeddings"""
        try:
            import json
            import numpy as np
            
            # Charger corpus
            if not CORPUS_PATH.exists():
                logger.error(f"❌ Corpus introuvable: {CORPUS_PATH}")
                raise FileNotFoundError(f"Corpus non trouvé: {CORPUS_PATH}")
            
            logger.info(f"📖 Chargement corpus: {CORPUS_PATH}")
            with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            logger.info(f"✅ Corpus chargé: {len(corpus)} documents")
            
            # Charger ou générer embeddings
            if EMBEDDINGS_PATH.exists():
                logger.info(f"📊 Chargement embeddings: {EMBEDDINGS_PATH}")
                embeddings = np.load(EMBEDDINGS_PATH)
                logger.info(f"✅ Embeddings chargés: shape {embeddings.shape}")
            else:
                logger.info("🔧 Génération des embeddings...")
                if not self.embedding_model:
                    self.embedding_model = EmbeddingPipeline(str(CORPUS_PATH))
                    self.embedding_model.initialize_embedding_model()
                
                # Générer embeddings avec batch size réduit pour économiser la mémoire
                texts = [doc.get('text', doc.get('contenu', '')) for doc in corpus]
                batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))  # Réduit par défaut
                embeddings = self.embedding_model.embedding_model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normaliser pour meilleure performance
                )
                
                # Sauvegarder embeddings
                np.save(EMBEDDINGS_PATH, embeddings)
                logger.info(f"✅ Embeddings générés et sauvegardés: {embeddings.shape}")
            
            # Normaliser corpus pour FAISS
            normalized_corpus = []
            for i, doc in enumerate(corpus):
                normalized_doc = {
                    'id': doc.get('id', doc.get('chunk_id', f'doc_{i}')),
                    'titre': doc.get('titre', doc.get('title', 'Document')),
                    'contenu': doc.get('text', doc.get('contenu', doc.get('content', ''))),
                    'source': doc.get('source', doc.get('source_institution', 'Unknown')),
                    'organisme': doc.get('organisme', doc.get('source_institution', 'Unknown')),
                    'type': doc.get('type', 'general')
                }
                normalized_corpus.append(normalized_doc)
            
            # Créer index FAISS
            logger.info("🏗️ Création index FAISS...")
            self.vector_store.create_index(normalized_corpus, embeddings)
            logger.info("✅ Index FAISS créé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur création index: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def answer_question(self, question: str, max_results: int = 3) -> Dict[str, Any]:
        """Répond à une question agricole"""
        if not self.initialized:
            return {
                "success": False,
                "reponse": "Système non initialisé. Veuillez patienter...",
                "sources": [],
                "error": "System not initialized"
            }
        
        try:
            logger.info(f"❓ Question: {question}")
            
            # 1. Générer embedding de la question
            question_embedding = self.embedding_model.embedding_model.encode([question])[0]
            
            # 2. Recherche vectorielle
            search_results = self.vector_store.search(question_embedding, k=max_results)
            
            if not search_results:
                return {
                    "success": False,
                    "reponse": "Je n'ai pas trouvé de documents pertinents dans ma base de connaissances pour répondre à cette question.",
                    "sources": []
                }
            
            # 3. Préparer contexte pour LLM
            context_docs = []
            sources_info = []
            
            for result in search_results:
                context_docs.append({
                    "text": result.document_text,
                    "metadata": result.metadata
                })
                sources_info.append({
                    "titre": result.metadata.get("titre", "Document"),
                    "source": result.metadata.get("source", "Unknown"),
                    "organisme": result.metadata.get("organisme", "Unknown"),
                    "pertinence": float(result.similarity_score)
                })
            
            # 4. Générer réponse avec LLM
            llm_response = self.llm_handler.generate_answer(
                question,
                context_docs,
                template=None  # Utiliser template par défaut
            )
            
            # 5. Construire réponse finale
            return {
                "success": llm_response.success,
                "reponse": llm_response.text,
                "sources": sources_info,
                "metadata": {
                    "backend": llm_response.backend,
                    "model": llm_response.model,
                    "generation_time": llm_response.generation_time,
                    "documents_used": len(search_results)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement question: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "reponse": f"Erreur lors du traitement: {str(e)}",
                "sources": [],
                "error": str(e)
            }

# ============================================================================
# INITIALISATION GLOBALE
# ============================================================================

rag_system = AgroConsultingRAG()

# Initialiser au démarrage
def init_system():
    """Initialise le système RAG"""
    return rag_system.initialize()

# ============================================================================
# INTERFACE GRADIO
# ============================================================================

import gradio as gr

def process_question(question: str, history: List[List[str]]) -> Tuple[List[List[str]], str]:
    """Traite une question et retourne la réponse"""
    if not question.strip():
        return history, ""
    
    # Ajouter question à l'historique
    history.append([question, None])
    
    # Obtenir réponse
    response = rag_system.answer_question(question, max_results=3)
    
    # Formater réponse
    if response["success"]:
        reponse_text = response["reponse"]
        
        # Ajouter sources si disponibles
        if response.get("sources"):
            reponse_text += "\n\n📚 **Sources:**\n"
            for i, source in enumerate(response["sources"][:3], 1):
                reponse_text += f"{i}. {source['titre']} ({source['organisme']}) - Pertinence: {source['pertinence']:.2f}\n"
    else:
        reponse_text = f"❌ Erreur: {response.get('reponse', 'Erreur inconnue')}"
    
    # Mettre à jour historique
    history[-1][1] = reponse_text
    
    return history, ""

# CSS personnalisé
CSS = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #2d5016 0%, #558b2f 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

# Créer l'interface Gradio
def create_interface():
    """Crée l'interface Gradio"""
    
    # En-tête
    header = gr.HTML("""
    <div class="main-header">
        <h1>🌾 AgroConsulting - Assistant IA Agricole</h1>
        <p>Posez vos questions sur l'agriculture au Burkina Faso</p>
    </div>
    """)
    
    # Chatbot
    chatbot = gr.Chatbot(
        label="💬 Chat",
        height=500,
        show_label=True,
        container=True,
        bubble_full_width=False
    )
    
    # Champ de saisie
    question_input = gr.Textbox(
        label="Votre question",
        placeholder="Ex: Quel engrais utiliser pour le mil en saison sèche ?",
        lines=2,
        max_lines=5
    )
    
    # Boutons
    with gr.Row():
        submit_btn = gr.Button("Envoyer 📤", variant="primary", scale=2)
        clear_btn = gr.Button("Effacer 🗑️", variant="secondary", scale=1)
    
    # État pour l'historique
    state = gr.State([])
    
    # Événements
    def submit(question, history):
        new_history, _ = process_question(question, history or [])
        return new_history, "", new_history
    
    def clear():
        return [], None, []
    
    submit_btn.click(
        fn=submit,
        inputs=[question_input, state],
        outputs=[chatbot, question_input, state]
    )
    
    question_input.submit(
        fn=submit,
        inputs=[question_input, state],
        outputs=[chatbot, question_input, state]
    )
    
    clear_btn.click(
        fn=clear,
        outputs=[chatbot, question_input, state]
    )
    
    # Interface
    interface = gr.Blocks(css=CSS, theme=gr.themes.Soft())
    
    with interface:
        header
        chatbot
        question_input
        with gr.Row():
            submit_btn
            clear_btn
        state
        gr.Markdown("""
        ### 📝 Exemples de questions:
        - Quel engrais utiliser pour le mil ?
        - Comment protéger le maïs des ravageurs ?
        - Quand planter le sorgho au Burkina Faso ?
        - Techniques de conservation des sols
        - Maladies courantes du riz
        """)
    
    return interface

# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

def main():
    """Fonction principale pour Hugging Face Spaces"""
    logger.info("🚀 Démarrage AgroConsulting sur Hugging Face Spaces")
    
    # Initialiser le système
    logger.info("⏳ Initialisation du système RAG (cela peut prendre quelques minutes)...")
    init_success = init_system()
    
    if not init_success:
        logger.error("❌ Échec initialisation, démarrage en mode dégradé")
        # Créer interface minimale en cas d'échec
        interface = gr.Interface(
            fn=lambda x: "❌ Système non disponible. Veuillez réessayer plus tard.",
            inputs="text",
            outputs="text",
            title="🌾 AgroConsulting - Système non disponible",
            description="Le système est en cours d'initialisation. Veuillez patienter..."
        )
    else:
        # Créer interface complète
        interface = create_interface()
    
    # Lancer l'interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()

