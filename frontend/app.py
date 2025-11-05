import gradio as gr
import requests
import json
from datetime import datetime
import logging
from typing import Tuple
import time

# ==================== CONFIGURATION ==================== #

TITRE = "AgroConsolling - Chat RAG Agricole"
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 60

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSS_CUSTOM = """
:root {
    --primary-green: #2d5016;
    --secondary-green: #558b2f;
    --light-green: #c5e1a5;
    --accent-yellow: #fbc02d;
}

* {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    box-sizing: border-box;
}

html, body {
    margin: 0;
    padding: 0;
    height: 100%;
}

.chat-container {
    display: flex;
    height: 100vh;
    background-color: white;
    border-radius: 0;
    overflow: hidden;
}

.sidebar {
    width: 260px;
    background: linear-gradient(135deg, #2d5016 0%, #558b2f 100%);
    color: white;
    padding: 20px;
    overflow-y: auto;
    box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    flex-shrink: 0;
}

.sidebar-title {
    font-size: 1.5em;
    font-weight: bold;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.btn-new-chat {
    width: 100%;
    padding: 12px;
    background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%);
    color: #2d5016;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    cursor: pointer;
    font-size: 0.95em;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}

.btn-new-chat:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(251, 192, 45, 0.3);
}

.status-indicator {
    padding: 12px;
    border-radius: 8px;
    font-size: 0.9em;
    margin-bottom: 15px;
    text-align: center;
    border-left: 4px solid;
}

.status-healthy {
    background: #e8f5e9;
    border-left-color: #558b2f;
    color: #2d5016;
}

.status-error {
    background: #ffebee;
    border-left-color: #c62828;
    color: #c62828;
}

.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    background-color: #ffffff;
    overflow: hidden;
    height: 100%;
}

.chat-header {
    padding: 20px;
    background: linear-gradient(135deg, #f5f5f5 0%, #e8f5e9 100%);
    border-bottom: 2px solid #c5e1a5;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
}

.header-title {
    font-size: 1.3em;
    color: #2d5016;
    font-weight: bold;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 15px;
    min-height: 0;
}

.message {
    display: flex;
    gap: 12px;
    margin-bottom: 10px;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.message.user {
    justify-content: flex-end;
}

.message.assistant {
    justify-content: flex-start;
}

.message-content {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 12px;
    word-wrap: break-word;
    line-height: 1.5;
}

.user-message {
    background: linear-gradient(135deg, #2d5016 0%, #558b2f 100%);
    color: white;
    border-bottom-right-radius: 4px;
}

.assistant-message {
    background: #f0f0f0;
    color: #333;
    border-bottom-left-radius: 4px;
    border: 1px solid #e0e0e0;
}

.sources-container {
    background: #f5f5f5;
    border-left: 4px solid #558b2f;
    padding: 12px;
    border-radius: 4px;
    margin-top: 8px;
    font-size: 0.9em;
}

.source-item {
    margin: 8px 0;
    padding: 10px;
    background: white;
    border-radius: 4px;
    border-left: 3px solid #c5e1a5;
}

.source-title {
    font-weight: bold;
    color: #2d5016;
    margin-bottom: 4px;
}

.source-org {
    font-size: 0.85em;
    color: #666;
    margin-bottom: 6px;
}

.pertinence-bar {
    background: #f0f0f0;
    height: 4px;
    border-radius: 2px;
    overflow: hidden;
}

.pertinence-fill {
    background: linear-gradient(90deg, #558b2f 0%, #7cb342 100%);
    height: 100%;
}

.metadata-info {
    font-size: 0.8em;
    color: #999;
    margin-top: 8px;
    padding: 8px;
    background: #fafafa;
    border-radius: 4px;
}

.loading-spinner {
    display: inline-block;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.error-message {
    background: #ffebee;
    border-left: 4px solid #c62828;
    color: #c62828;
    padding: 12px;
    border-radius: 4px;
    margin-bottom: 10px;
}

.success-message {
    background: #e8f5e9;
    border-left: 4px solid #558b2f;
    color: #2d5016;
    padding: 12px;
    border-radius: 4px;
    margin-bottom: 10px;
}

.input-area {
    padding: 15px;
    background: white;
    border-top: 1px solid #e0e0e0;
    flex-shrink: 0;
    display: flex;
    align-items: flex-end;
    gap: 10px;
    min-height: auto;
}

.input-field {
    flex: 1;
    padding: 12px;
    border: 2px solid #c5e1a5;
    border-radius: 8px;
    font-size: 0.95em;
    transition: all 0.3s ease;
    resize: vertical;
    max-height: 120px;
    min-height: 45px;
}

.input-field:focus {
    outline: none;
    border-color: #558b2f;
    box-shadow: 0 0 0 3px rgba(85, 139, 47, 0.1);
}

.btn-send {
    padding: 12px 20px;
    background: linear-gradient(135deg, #2d5016 0%, #558b2f 100%);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: bold;
    transition: all 0.3s ease;
    font-size: 0.8em;
    display: flex;
    align-items: center;
    justify-content: center;
    white-space: nowrap;
    flex-shrink: 0;
    height: 45px;
}

.btn-send:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(45, 80, 22, 0.3);
}

.btn-send:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #999;
    text-align: center;
}

.empty-state-icon {
    font-size: 4em;
    margin-bottom: 20px;
}

.empty-state-text {
    font-size: 1.1em;
    color: #666;
}
"""

JS_SCROLL = """
<script>
function scrollToBottom() {
    const chatBox = document.querySelector('[id*="chat-display"]');
    if (chatBox) {
        setTimeout(() => {
            chatBox.scrollTop = chatBox.scrollHeight;
        }, 100);
    }
}
window.addEventListener('load', scrollToBottom);
</script>
"""

# ==================== CLASSE API CLIENT ==================== #

class AgroConsollingAPIClient:
    """Client pour l'API RAG Agricole"""
    
    def __init__(self, api_url: str = API_BASE_URL, timeout: int = API_TIMEOUT):
        self.api_url = api_url
        self.timeout = timeout
    
    def check_health(self) -> Tuple[bool, str]:
        """Vérifie l'état du backend API"""
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                
                if status == "healthy" or status == "degraded":
                    components = data.get("components", {})
                    llm_status = components.get("llm_handler", {}).get("status", "unknown")
                    if status == "healthy":
                        return True, f"✅ Système prêt (LLM: {llm_status})"
                    else:
                        return True, f"⚠️ Mode dégradé mais fonctionnel"
                else:
                    return False, f"❌ Statut: {status}"
            else:
                return False, f"❌ HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "❌ Impossible de se connecter à l'API (vérifiez port 8000)"
        except requests.exceptions.Timeout:
            return False, "❌ Timeout API"
        except Exception as e:
            return False, f"❌ Erreur: {str(e)[:50]}"
    
    def ask_question(
        self,
        question: str,
        max_results: int = 3,
        template: str = "standard"
    ) -> dict:
        """Envoie une question à l'API"""
        try:
            payload = {
                "question": question,
                "max_results": max_results,
                "template": template,
                "verbose": False
            }
            
            response = requests.post(
                f"{self.api_url}/ask",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "details": response.text[:200]
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Timeout",
                "details": "Réponse > 60s. Le modèle LLM peut être lent."
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connexion échouée",
                "details": f"Impossible de joindre {self.api_url}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": "Erreur",
                "details": str(e)[:100]
            }

# ==================== CLIENT GLOBAL ==================== #

api_client = AgroConsollingAPIClient()

# ==================== FONCTIONS DE TRAITEMENT ==================== #

def format_response_with_sources(api_response: dict) -> str:
    """Formate la réponse avec sources et métadonnées"""
    
    if not api_response.get("success", False):
        error = api_response.get("error", "Erreur inconnue")
        details = api_response.get("details", "")
        return f"""<div class="error-message">
            <strong>❌ Erreur:</strong> {error}<br/>
            <small>{details}</small>
        </div>"""
    
    # Réponse principale
    html = f"""<div class="message-content assistant-message" style="background: #e8f5e9; border-left: 4px solid #558b2f; padding: 15px; border-radius: 8px;">
        {api_response.get('reponse', 'Aucune réponse')}
    </div>"""
    
    # Sources
    sources = api_response.get("sources", [])
    if sources:
        html += """<div class="sources-container">
            <strong>📚 Sources utilisées:</strong>"""
        
        for i, source in enumerate(sources, 1):
            pertinence = source.get("pertinence", 0)
            titre = source.get("titre", "Inconnue")
            organisme = source.get("organisme", "N/A")
            bar_width = int(pertinence * 100)
            
            html += f"""
            <div class="source-item">
                <div class="source-title">{i}. {titre}</div>
                <div class="source-org">{organisme}</div>
                <div class="pertinence-bar">
                    <div class="pertinence-fill" style="width: {bar_width}%"></div>
                </div>
                <small>Pertinence: {pertinence:.0%}</small>
            </div>"""
        
        html += "</div>"
    
    # Métadonnées
    metadata = api_response.get("metadata", {})
    if metadata:
        gen_time = metadata.get("generation_time", 0)
        tokens = metadata.get("tokens_generated", 0)
        proc_time = metadata.get("processing_time", 0)
        model = metadata.get("model", "Unknown")
        docs = metadata.get("documents_used", 0)
        
        html += f"""<div class="metadata-info">
            ⏱️ {proc_time:.2f}s | 🤖 {model} | 📊 {tokens} tokens | 📄 {docs} doc
        </div>"""
    
    return html

def process_chat_message(message: str, history: list, api_available: bool) -> Tuple[str, list, str]:
    """Traite un message utilisateur"""
    
    if not message or not message.strip():
        return "", history, ""
    
    if not api_available:
        error_html = """<div class="error-message">
            <strong>❌ API non disponible</strong><br/>
            Vérifiez: uvicorn api.main:app --reload
        </div>"""
        return error_html, history, ""
    
    # Ajouter à l'historique (user)
    history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    # Appeler API
    logger.info(f"Envoi question: {message[:50]}...")
    api_response = api_client.ask_question(message, max_results=3, template="standard")
    
    # Formater réponse
    formatted_response = format_response_with_sources(api_response)
    
    # Ajouter à l'historique (assistant)
    history.append({
        "role": "assistant",
        "content": formatted_response,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    # Construire HTML
    chat_html = build_chat_html(history)
    
    return chat_html, history, ""

def build_chat_html(history: list) -> str:
    """Construit le HTML du chat"""
    if not history:
        return """
        <div class="empty-state">
            <div class="empty-state-icon">🌾</div>
            <div class="empty-state-text">Bienvenue sur AgroConsolling!</div>
            <p style="color:#ccc;margin-top:10px;">Posez une question pour commencer</p>
        </div>
        """
    
    html = ""
    for msg in history:
        if msg["role"] == "user":
            html += f"""
            <div class="message user">
                <div style="max-width: 75%;">
                    <div class="message-content user-message">{msg["content"]}</div>
                </div>
            </div>"""
        else:
            html += f"""
            <div class="message assistant">
                <div style="max-width: 75%;">
                    {msg["content"]}
                </div>
            </div>"""
    
    return html

def new_conversation() -> Tuple[str, list]:
    """Crée une nouvelle conversation"""
    empty_html = """
    <div class="empty-state">
        <div class="empty-state-icon">🌾</div>
        <div class="empty-state-text">Nouvelle conversation créée</div>
        <p style="color:#ccc;margin-top:10px;">Posez votre première question</p>
    </div>
    """
    return empty_html, []

# ==================== INTERFACE GRADIO ==================== #

with gr.Blocks(title=TITRE, css=CSS_CUSTOM + JS_SCROLL, theme=gr.themes.Soft()) as demo:
    
    # État
    chat_history = gr.State(value=[])
    api_status = gr.State(value={"available": False, "message": "Vérification..."})
    
    with gr.Row(elem_classes="chat-container"):
        # Sidebar
        with gr.Column(scale=1, min_width=260, elem_classes="sidebar"):
            gr.HTML('<div class="sidebar-title">🌾 AgroConsolling</div>')
            
            # Statut API
            status_display = gr.HTML()
            
            btn_new = gr.Button(
                "➕ Nouvelle conversation", size="lg", elem_classes="btn-new-chat"
            )
            
            gr.HTML(
                '<div style="margin-top:20px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.2);font-weight:bold;">Historique</div>'
            )
            gr.HTML(
                '<div style="margin-bottom: 20px;"><div style="padding: 12px; background: rgba(255,255,255,0.1); border-radius: 8px; border-left: 3px solid #fbc02d; font-weight: bold;">💬 Conversation actuelle</div></div>'
            )
            
            gr.HTML(
                '<div style="margin-top:30px;padding-top:20px;border-top:1px solid rgba(255,255,255,0.2);font-size:0.85em;opacity:0.8;">À propos<br/>AgroConsolling v2.0<br/>Powered by RAG + Ollama</div>'
            )
        
        # Main Chat Area
        with gr.Column(scale=4, elem_classes="main-content"):
            gr.HTML(
                '<div class="chat-header"><div class="header-title">🧑🏾‍🌾 Chat AgroConsolling RAG</div></div>'
            )
            
            chat_display = gr.HTML(
                value="""
                <div class="empty-state">
                    <div class="empty-state-icon">🌾</div>
                    <div class="empty-state-text">Bienvenue sur AgroConsolling!</div>
                    <p style="color:#ccc;margin-top:10px;">Posez une question agricole pour commencer</p>
                </div>
                """,
                elem_id="chat-display",
                elem_classes="chat-messages",
            )
            
            with gr.Group(elem_classes="input-area"):
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ex: Quel engrais pour le mil en saison sèche ?",
                        show_label=False,
                        lines=1,
                        max_lines=4,
                        elem_classes="input-field",
                        scale=4,
                    )
                    btn_send = gr.Button(
                        "➤", scale=1, elem_classes="btn-send", visible=False
                    )
    
    # ==================== ÉVÉNEMENTS ==================== #
    
    def toggle_send_btn(message):
        return gr.update(visible=bool(message.strip()))
    
    def check_api_on_load():
        is_available, message = api_client.check_health()
        status_html = f"""<div class="status-indicator {'status-healthy' if is_available else 'status-error'}">
            {message}
        </div>"""
        return status_html, {"available": is_available, "message": message}
    
    def handle_send(message, history, status_dict):
        chat_html, new_history, _ = process_chat_message(
            message, 
            history, 
            status_dict["available"]
        )
        return chat_html, new_history, ""
    
    # Changements
    msg_input.change(fn=toggle_send_btn, inputs=[msg_input], outputs=[btn_send])
    
    # Soumission
    msg_input.submit(
        fn=handle_send,
        inputs=[msg_input, chat_history, api_status],
        outputs=[chat_display, chat_history, msg_input],
    )
    
    btn_send.click(
        fn=handle_send,
        inputs=[msg_input, chat_history, api_status],
        outputs=[chat_display, chat_history, msg_input],
    )
    
    # Nouvelle conversation
    btn_new.click(
        fn=lambda: (new_conversation()[0], new_conversation()[1]),
        inputs=[],
        outputs=[chat_display, chat_history],
    )
    
    # Check API au chargement
    demo.load(
        fn=check_api_on_load,
        inputs=[],
        outputs=[status_display, api_status],
    )

# ==================== LANCEMENT ==================== #

if __name__ == "__main__":
    logger.info("🚀 Démarrage AgroConsolling RAG Frontend")
    logger.info(f"📡 API URL: {API_BASE_URL}")
    demo.launch(share=False, show_error=True)