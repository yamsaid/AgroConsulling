![header](https://capsule-render.vercel.app/api?type=cylinder&color=0:0f766e,100:065f46&height=180&text=AgroConsulling&fontSize=30&fontColor=ffffff&desc=Assistant%20IA%20Agricole%20pour%20le%20Burkina%20Faso%20|%20Hackathon%20IA%202025&descSize=15&descAlignY=75)

<p align="center">

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>

<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>

<img src="https://img.shields.io/badge/Gradio-Frontend-FF6F00?style=for-the-badge&logo=gradio&logoColor=white"/>

<img src="https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-blueviolet?style=for-the-badge"/>

<img src="https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge"/>

<img src="https://img.shields.io/badge/Open%20Source-100%25-success?style=for-the-badge"/>

<img src="https://img.shields.io/badge/Burkina%20Faso-Agriculture-green?style=for-the-badge"/>

</p>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

<p align="center">
<a href="#">
<img src="https://img.shields.io/badge/🇫🇷%20Français-2d6a4f?style=for-the-badge"/>
</a>

<a href="README_EN.md">
<img src="https://img.shields.io/badge/🇬🇧%20English-1d3557?style=for-the-badge"/>
</a>
</p>

# Résumé

*AgroConsulling est un assistant conversationnel intelligent conçu pour démocratiser l'accès à l'information agricole au Burkina Faso. Basé sur une architecture Retrieval-Augmented Generation (RAG), il permet aux agriculteurs, techniciens, étudiants et décideurs d'obtenir des recommandations agricoles contextualisées, fiables et traçables à partir d'une base documentaire spécialisée.*

*Le projet a été développé dans le cadre du Hackathon des Universités en Intelligence Artificielle avec l'objectif de démontrer qu'il est possible de construire une solution d'IA performante, souveraine et entièrement open source répondant aux besoins réels du monde agricole burkinabè.*

### 🚀 Principaux résultats

✔ Plus de **500 documents agricoles** intégrés dans la base de connaissances

✔ Architecture **100 % Open Source**

✔ Assistant IA spécialisé en agriculture burkinabè

✔ Pipeline RAG complet avec recherche vectorielle

✔ Réponses contextualisées avec citations des sources

✔ Temps moyen de réponse inférieur à **2 secondes**

✔ Interface web responsive adaptée aux appareils mobiles

✔ Déploiement local sans dépendance à des services propriétaires

**Compétences mobilisées :** Intelligence Artificielle, RAG, NLP, FastAPI, Gradio, Vector Search, Open Source AI, Data Engineering, Développement API.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 📌 Contexte et justification

L'agriculture représente le principal pilier économique du Burkina Faso et mobilise près de **86 % de la population active**. Malgré cette importance stratégique, les producteurs agricoles rencontrent de nombreuses difficultés liées à l'accès à une information technique fiable et actualisée.

Les connaissances agricoles existent sous diverses formes :

- Guides techniques de la FAO
- Rapports du Ministère de l'Agriculture
- Publications de l'INSD
- Études du CIRAD
- Rapports de projets et ONG

Cependant, ces ressources demeurent souvent dispersées, difficiles d'accès ou peu exploitables directement par les producteurs.

> 💡 **Problématique**
>
> Comment permettre aux agriculteurs burkinabè d'accéder rapidement à des conseils agricoles fiables, contextualisés et basés sur des sources documentaires reconnues ?

AgroConsulling répond à cette problématique grâce à un assistant conversationnel alimenté par l'intelligence artificielle et une base documentaire spécialisée.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🎯 Objectifs

### Objectif principal

Démocratiser l'accès à l'information technique agricole grâce à un assistant IA spécialisé.

### Objectifs spécifiques

- Centraliser la documentation agricole de référence.
- Faciliter l'accès aux connaissances agronomiques.
- Fournir des recommandations contextualisées.
- Promouvoir la souveraineté technologique grâce à l'open source.
- Accompagner les producteurs dans leurs prises de décision.
- Valoriser les données et connaissances agricoles nationales.
- Développer une solution reproductible dans d'autres secteurs.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🏗️ Architecture technique

```text
Question utilisateur
        │
        ▼
 Interface Gradio
        │
        ▼
 API FastAPI
        │
        ▼
 Génération Embedding
 Sentence Transformers
        │
        ▼
 Recherche Vectorielle
 ChromaDB / FAISS
        │
        ▼
 Documents pertinents
        │
        ▼
 LLM Local
 Mistral / Llama 3.2
        │
        ▼
 Réponse contextualisée
 avec sources
```

### Schéma du système

![Architecture](image.png)

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🛠️ Technologies Open Source utilisées

<p align="center">

<img src="https://img.shields.io/badge/Python-Langage-3776AB?style=flat-square"/>

<img src="https://img.shields.io/badge/FastAPI-API-009688?style=flat-square"/>

<img src="https://img.shields.io/badge/Gradio-Frontend-orange?style=flat-square"/>

<img src="https://img.shields.io/badge/Ollama-LLM-black?style=flat-square"/>

<img src="https://img.shields.io/badge/Mistral-7B-success?style=flat-square"/>

<img src="https://img.shields.io/badge/Llama3.2-LLM-blue?style=flat-square"/>

<img src="https://img.shields.io/badge/FAISS-Vector%20Store-purple?style=flat-square"/>

<img src="https://img.shields.io/badge/ChromaDB-Database-red?style=flat-square"/>

<img src="https://img.shields.io/badge/SentenceTransformers-Embeddings-green?style=flat-square"/>

</p>

| Technologie           | Rôle                     | Licence     |
| --------------------- | ------------------------ | ----------- |
| Python                | Développement principal  | PSF         |
| FastAPI               | Backend API              | MIT         |
| Gradio                | Interface utilisateur    | Apache 2.0  |
| Ollama                | Exécution locale des LLM | MIT         |
| Sentence Transformers | Génération d'embeddings  | Apache 2.0  |
| ChromaDB              | Base vectorielle         | Apache 2.0  |
| FAISS                 | Recherche vectorielle    | MIT         |
| Mistral 7B            | Modèle de langage        | Apache 2.0  |
| Llama 3.2             | Modèle de génération     | Open Source |

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 📦 Installation

## Prérequis

| Ressource     | Minimum          |
| ------------- | ---------------- |
| Python        | 3.8+             |
| RAM           | 8 Go             |
| Espace disque | 4 Go             |
| Git           | Dernière version |
| Ollama        | Installé         |

### 1. Cloner le dépôt

```bash
git clone https://github.com/yamsaid/AgroConsulling.git

cd AgroConsulling
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

Activation :

```bash
source venv/Scripts/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Installer Ollama

Linux / Mac :

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Windows :

Télécharger l'installateur depuis :

[https://ollama.com](https://ollama.com)

### 5. Télécharger les modèles

```bash
ollama pull mistral
```

```bash
ollama pull llama3.2:3b
```

### 6. Vérifier l'installation

```bash
ollama run mistral
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# ▶️ Démarrage

### Lancer l'API

```bash
uvicorn src.api:app
```

ou

```bash
python -m uvicorn src.api:app
```

### Lancer l'interface

```bash
python frontend/app.py
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 📊 Évaluation du système

<table>

<tr>

<td width="50%" valign="top">

<h3>Performance</h3>

| Indicateur            | Valeur        |
| --------------------- | ------------- |
| Corpus                | 503 documents |
| Questions de test     | 20            |
| Temps moyen           | ~2 secondes   |
| Sources citées        | Oui           |
| Recherche vectorielle | Fonctionnelle |

</td>

<td width="50%" valign="top">

<h3>Validation</h3>

| Critère                  | Résultat      |
| ------------------------ | ------------- |
| Pertinence des réponses  | Satisfaisante |
| Adaptation contexte BF   | Oui           |
| Conseils agricoles       | Validés       |
| Références documentaires | Présentes     |
| Pipeline RAG             | Opérationnel  |

</td>

</tr>

</table>

> [!TIP]
>
> Le système a été évalué à partir d'un ensemble de 20 questions agricoles couvrant le mil, le sorgho, le maïs, le maraîchage et les bonnes pratiques agricoles adaptées au contexte burkinabè.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🖥️ Aperçu de l'application

### Interface utilisateur

![Interface](AJOUTER_IMAGE_INTERFACE)

### Écran principal du chatbot

![Chatbot](AJOUTER_CAPTURE_CHATBOT)

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 📁 Structure du projet

```text
📦 AgroConsulling
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── corpus.json
│   └── sources.txt
│
├── src/
│   ├── api.py
│   ├── rag_pipeline.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── llm_handler.py
│
├── frontend/
│   └── app.py
│
└── evaluation/
    └── evaluate.py
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 🌍 Impact attendu

* Renforcement des capacités techniques des producteurs.
* Diffusion des bonnes pratiques agricoles.
* Amélioration de la productivité agricole.
* Réduction de l'asymétrie d'information.
* Valorisation des connaissances locales.
* Contribution à la souveraineté numérique du Burkina Faso.
* Réplicabilité dans d'autres secteurs du développement.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# ⚠️ Limites

> [!WARNING]
>
> Les performances du système dépendent de la qualité et de la couverture documentaire du corpus utilisé.

> [!WARNING]
>
> Les recommandations fournies constituent une aide à la décision et ne remplacent pas l'expertise agronomique de terrain.

> [!WARNING]
>
> Certaines problématiques très spécifiques peuvent nécessiter un enrichissement futur de la base documentaire.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 👥 Équipe

<table align="center">

<tr>

<td align="center">

<b>YAMEOGO Saïdou</b><br/> <sub>Data Scientist</sub><br/> <a href="https://github.com/yamsaid"> <img src="https://img.shields.io/badge/GitHub-yamsaid-181717?style=flat-square&logo=github"/> </a>

</td>

<td align="center">

<b>SANOU Ange Noëlie</b><br/> <sub>Data Scientist</sub>

</td>

<td align="center">

<b>NIAMPA Abdoul Fataho</b><br/> <sub>Data Scientist</sub>

</td>

</tr>

</table>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:4facfe,100:00f2fe&height=3" width="100%"/>

# 📚 Références

* FAO (2023) — Guides techniques agricoles.
* INSD (2023) — Enquête Permanente Agricole.
* Banque Mondiale (2022) — Diagnostic du secteur agricole.
* CIRAD (2023) — Innovations agricoles.
* Programme National du Secteur Rural.
* GIEC (2022) — Impacts climatiques en Afrique de l'Ouest.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f766e,100:065f46&height=100&section=footer"/>