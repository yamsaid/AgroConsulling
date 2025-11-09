import gradio as gr
import requests
import json
from datetime import datetime
import logging
from typing import Tuple
import sys

# ==================== CONFIGURATION ==================== #

TITRE = "AgroConsulling - Chat RAG Agricole"
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 600

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
sys.stdout.reconfigure(encoding="utf-8")

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

/* ===== SIDEBAR ===== */
.sidebar {
    width: 500px;
    background: linear-gradient(135deg, #2d5016 0%, #558b2f 100%);
    color: white;
    padding: 30px 20px;
    overflow-y: auto;
    box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    flex-shrink: 0;
    min-height: 100vh;
    height: 100%;
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

/* ===== HISTORIQUE ===== */
.history-list {
    margin-top: 10px;
}

.history-item {
    padding: 12px;
    margin: 8px 0;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.history-item:hover {
    background: rgba(255, 255, 255, 0.2);
}

.history-item.active {
    background: rgba(251, 192, 45, 0.2);
    border-left: 3px solid var(--accent-yellow);
}

.history-title {
    font-size: 0.9em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-date {
    font-size: 0.8em;
    color: rgba(255, 255, 255, 0.6);
}

/* ===== ZONE PRINCIPALE ===== */
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

/* ===== CHAT ===== */
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
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

.message.user { justify-content: flex-end; }
.message.assistant { justify-content: flex-start; }

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

/* ===== SPINNER DE CHARGEMENT ===== */
.loading-spinner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 15px;
    background: #e8f5e9;
    border-radius: 8px;
    border-left: 4px solid #558b2f;
    margin: 10px 0;
}

.spinner-icon {
    font-size: 24px;
    animation: spin 2s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.spinner-text {
    color: #2d5016;
    font-weight: 500;
}

/* ===== SOURCES ===== */
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

/* ===== CHAMP DE SAISIE ===== */
#input-zone {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    padding: 15px 0;
    background: white;
    border-top: 1px solid #e0e0e0;
}

.input-wrapper {
    display: flex;
    align-items: stretch;
    width: 95%;
    max-width: 1200px;
    min-width: 600px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    overflow: hidden;
}

.input-field textarea {
    border: none !important;
    padding: 14px 18px !important;
    font-size: 1rem !important;
    width: 100% !important;
    resize: none !important;
    min-height: 50px !important;
}

.btn-send {
    background: linear-gradient(135deg, #2d5016 0%, #558b2f 100%);
    color: white;
    border: none;
    font-size: 20px;
    padding: 0 20px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
}

.btn-send:hover {
    background: #33691e;
}

/* ===== MASQUER LA MINUTERIE GRADIO ===== */
.eta-bar, .progress-bar, .generating {
    display: none !important;
    visibility: hidden !important;
}

/* ===== FOOTER ===== */
footer, .svelte-1ipelgc, .svelte-1y4p8pa {
    display: none !important;
    visibility: hidden !important;
}
"""

JS_CUSTOM = """
<script>
// Fonction de défilement vers le bas du chat
function scrollToBottom() {
    const chatBox = document.querySelector('[id*="chat-display"]');
    if (chatBox) {
        setTimeout(() => {
            chatBox.scrollTop = chatBox.scrollHeight;
        }, 100);
    }
}

// Gestion des clics sur l'historique
function setupHistoryClickHandlers() {
    document.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', function() {
            const convId = this.id.split('-')[1];
            const btnId = `button#conv-btn-${convId}`;
            const btn = document.querySelector(btnId);
            if (btn) btn.click();
        });
    });
}

// Masquer les barres de progression Gradio
function hideProgressBars() {
    const selectors = [
        '.eta-bar',
        '.progress-bar', 
        '.generating',
        '[class*="progress"]',
        '[class*="eta"]'
    ];
    
    selectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => {
            el.style.display = 'none';
        });
    });
}

// Configuration initiale
window.addEventListener('load', () => {
    scrollToBottom();
    setupHistoryClickHandlers();
    hideProgressBars();
});

// Observer pour masquer les barres de progression
const progressObserver = new MutationObserver(() => {
    hideProgressBars();
});

// Observer pour les clics historique
const historyObserver = new MutationObserver(() => {
    setupHistoryClickHandlers();
});

// Démarrer les observations
window.addEventListener('load', () => {
    // Observer le body pour les barres de progression
    progressObserver.observe(document.body, { 
        childList: true, 
        subtree: true 
    });
    
    // Observer l'historique
    const historyList = document.querySelector('.history-list');
    if (historyList) {
        historyObserver.observe(historyList, { 
            childList: true, 
            subtree: true 
        });
    }
});
</script>
"""

# ==================== GESTION DE L'HISTORIQUE ==================== #


class ConversationManager:
    def __init__(self):
        self.conversations = []
        self.current_id = None

    def create_conversation(self) -> dict:
        conversation = {
            "id": len(self.conversations),
            "title": "Nouvelle conversation",
            "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "messages": [],
        }
        self.conversations.append(conversation)
        self.current_id = conversation["id"]
        return conversation

    def add_message(self, message: str, response: dict):
        if not self.conversations or self.current_id is None:
            self.create_conversation()

        current_conv = self.conversations[self.current_id]
        current_conv["messages"].append({"role": "user", "content": message})
        current_conv["messages"].append({"role": "assistant", "content": response})

        if len(current_conv["messages"]) == 2:
            clean_title = message.replace("\n", " ").replace("\r", " ")
            clean_title = (
                clean_title[:30] + "..." if len(clean_title) > 30 else clean_title
            )
            current_conv["title"] = clean_title

    def get_conversation(self, conv_id: int) -> dict:
        return (
            self.conversations[conv_id]
            if 0 <= conv_id < len(self.conversations)
            else None
        )

    def get_all_conversations(self) -> list:
        return self.conversations


conversation_manager = ConversationManager()

# ==================== CLASSE API CLIENT ==================== #


class AgroConsollingAPIClient:
    def __init__(self, api_url: str = API_BASE_URL, timeout: int = API_TIMEOUT):
        self.api_url = api_url
        self.timeout = timeout

    def ask_question(
        self, question: str, max_results: int = 3, template: str = "standard"
    ) -> dict:
        try:
            payload = {
                "question": question,
                "max_results": max_results,
                "template": template,
                "verbose": False,
            }
            response = requests.post(
                f"{self.api_url}/ask", json=payload, timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "details": response.text[:200],
                }
        except Exception as e:
            return {"success": False, "error": "Erreur", "details": str(e)[:100]}


api_client = AgroConsollingAPIClient()

# ==================== TRAITEMENT DU CHAT ==================== #


def format_response_with_sources(api_response: dict) -> str:
    if not api_response.get("success", False):
        return f"""<div class="error-message"><strong>❌ Erreur:</strong> {api_response.get("error", "Erreur inconnue")}<br/><small>{api_response.get("details", "")}</small></div>"""

    # Réponse principale
    html = f"""<div class="message-content assistant-message" 
                style="background: #e8f5e9; border-left: 4px solid #558b2f; padding: 15px; border-radius: 8px;">
                {api_response.get("reponse", "Aucune réponse disponible.")}
            </div>"""

    # Sources
    sources = api_response.get("sources", [])
    if sources:
        html += '<div class="sources-container"><strong>📚 Sources utilisées :</strong>'
        for i, source in enumerate(sources, 1):
            source_url = source.get("source_url") or source.get("url", "#")
            organisme = (
                source.get("source_institution")
                or source.get("organisme")
                or "Source inconnue"
            )

            html += f"""
            <div class="source-item">
                <div class="source-title">
                    <strong>🔗 {organisme}</strong>
                </div>
            </div>"""
        html += "</div>"
    return html


def show_loading_spinner() -> str:
    """Affiche le spinner de chargement"""
    return """
    <div class="loading-spinner">
        <div class="spinner-icon">🌾</div>
        <div class="spinner-text">Génération de la réponse en cours...</div>
    </div>
    """


def process_chat_message(message: str, history: list) -> Tuple[str, list, str]:
    if not message.strip():
        return "", history, ""

    # Ajouter le message utilisateur
    history.append({"role": "user", "content": message})

    # Afficher le spinner pendant le traitement
    temp_html = build_chat_html(history) + show_loading_spinner()
    yield temp_html, history, ""

    # Appeler l'API
    api_response = api_client.ask_question(message)
    formatted_response = format_response_with_sources(api_response)
    history.append({"role": "assistant", "content": formatted_response})

    # Mettre à jour l'historique
    conversation_manager.add_message(message, formatted_response)

    # Retourner la réponse finale
    yield build_chat_html(history), history, ""


def build_chat_html(history: list) -> str:
    if not history:
        return """<div class="empty-state"><div class="empty-state-icon">🌾</div><div class="empty-state-text">Bienvenue sur AgroConsulling!</div><p style="color:#ccc;margin-top:10px;">Posez une question pour commencer</p></div>"""
    html = ""
    for msg in history:
        if msg["role"] == "user":
            html += f"""<div class="message user"><div style="max-width: 75%;"><div class="message-content user-message">{msg["content"]}</div></div></div>"""
        else:
            html += f"""<div class="message assistant"><div style="max-width: 75%;">{msg["content"]}</div></div>"""
    return html


def new_conversation() -> Tuple[str, list, str]:
    conversation_manager.create_conversation()
    return (
        """<div class="empty-state"><div class="empty-state-icon">🌾</div><div class="empty-state-text">Nouvelle conversation créée</div><p style="color:#ccc;margin-top:10px;">Posez votre première question</p></div>""",
        [],
        gr.update(choices=get_radio_choices(), value=None),
    )


def get_radio_choices():
    return [conv["title"] for conv in conversation_manager.get_all_conversations()]


def switch_conversation(selected_title):
    for conv in conversation_manager.get_all_conversations():
        if conv["title"] == selected_title:
            conversation_manager.current_id = conv["id"]
            history = conv["messages"]
            return build_chat_html(history), history
    return None, None


# ==================== INTERFACE GRADIO ==================== #

with gr.Blocks(title=TITRE, css=CSS_CUSTOM + JS_CUSTOM, theme=gr.themes.Soft()) as demo:
    chat_history = gr.State(value=[])

    with gr.Row(elem_classes="chat-container"):
        # Sidebar
        with gr.Column(scale=1, min_width=260, elem_classes="sidebar"):
            gr.HTML('<div class="sidebar-title">🌾 AgroConsulling</div>')
            btn_new = gr.Button(
                "➕ Nouvelle conversation", size="lg", elem_classes="btn-new-chat"
            )
            gr.HTML(
                '<div style="margin-top:20px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.2);font-weight:bold;">Historique</div>'
            )
            radio_history = gr.Radio(
                choices=[
                    conv["title"]
                    for conv in conversation_manager.get_all_conversations()
                ],
                label="Conversations",
                value=None,
                elem_id="radio-history",
            )

        # Zone principale
        with gr.Column(scale=4, elem_classes="main-content"):
            gr.HTML(
                '<div class="chat-header"><div class="header-title">🧑🏾‍🌾 Chat AgroConsulling</div></div>'
            )
            chat_display = gr.HTML(
                value="""<div class="empty-state"><div class="empty-state-icon">🌾</div><div class="empty-state-text">Bienvenue sur AgroConsulling!</div><p style="color:#ccc;margin-top:10px;">Posez une question agricole pour commencer</p></div>""",
                elem_id="chat-display",
                elem_classes="chat-messages",
            )

            # Zone de saisie
            with gr.Group(elem_id="input-zone"):
                with gr.Row(elem_classes="input-wrapper"):
                    msg_input = gr.Textbox(
                        placeholder="Posez une question sur l'agriculture...",
                        show_label=False,
                        lines=1,
                        max_lines=4,
                        elem_classes="input-field",
                        scale=4,
                    )
                    btn_send = gr.Button("➤", elem_classes="btn-send", visible=False)

    # Événements
    def toggle_send_btn(message):
        return gr.update(visible=bool(message.strip()))

    msg_input.change(fn=toggle_send_btn, inputs=[msg_input], outputs=[btn_send])

    msg_input.submit(
        fn=process_chat_message,
        inputs=[msg_input, chat_history],
        outputs=[chat_display, chat_history, msg_input],
    ).then(fn=lambda: gr.update(choices=get_radio_choices()), outputs=[radio_history])

    btn_send.click(
        fn=process_chat_message,
        inputs=[msg_input, chat_history],
        outputs=[chat_display, chat_history, msg_input],
    ).then(fn=lambda: gr.update(choices=get_radio_choices()), outputs=[radio_history])

    btn_new.click(
        fn=new_conversation, outputs=[chat_display, chat_history, radio_history]
    )

    radio_history.change(
        fn=switch_conversation,
        inputs=[radio_history],
        outputs=[chat_display, chat_history],
    )

if __name__ == "__main__":
    logger.info("🚀 Démarrage AgroConsulling RAG Frontend")
    demo.launch(share=False, show_error=True)
