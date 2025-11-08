---
title: "README"
output: html_document
date: "2025-11-04"
---


![alt text](<WhatsApp Image 2025-11-05 à 22.49.41_1c24589a.jpg>)


# Plan

## Sujet choisi et justification 
## Architecture technique 
## Technologies open source utilisées (avec liens vers licences) 
## Instructions installation 
## Résultats évaluation



# AgroConsolling - Assistant IA Agricole
_(juste en dessous des badges sympatiques à placer)_

# 🌱 AgriConseil-BF - Assistant IA pour l'Agriculture Burkinabè

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open Source](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://opensource.org/)
[![Made in Burkina Faso](https://img.shields.io/badge/Made%20in-Burkina%20Faso-green.svg)](https://en.wikipedia.org/wiki/Burkina_Faso)
[![100% Open Source](https://img.shields.io/badge/Open%20Source-100%25-brightgreen.svg)]()
[![AI Powered](https://img.shields.io/badge/AI-Powered-ff69b4.svg)]()


## Description du projet (Contexte et justification)

Au Burkina Faso, l’agriculture constitue le principal moteur économique et social, occupant près de 86 % de la population active en 2025. Toutefois, les petits exploitants, particulièrement dans les zones périurbaines, rencontrent une difficulté majeure : le manque d’accès à une information technique, fiable et disponible à temps. Cette lacune, qu’elle concerne les pratiques culturales, la gestion des ravageurs, l’adaptation climatique ou les données de marché, limite fortement la productivité et les revenus agricoles.

**AgroConsolling** est une solution numérique innovante développée pour pallier ce déficit d’information. Il s’agit d’un **assistant virtuel intelligent** qui accompagne les agriculteurs, les techniciens et les étudiants du secteur dans leurs prises de décision. En quelques interactions, l’utilisateur peut obtenir des **recommandations adaptées à son contexte** sur la gestion des cultures, l’irrigation, la protection phytosanitaire, les pratiques durables ou encore la planification saisonnière.

AgroConsolling vise ainsi à permettre à tout acteur agricole — débutant ou expérimenté — d’élaborer et de simuler un projet complet : besoins en intrants, surfaces, investissements, calendrier de production, rendement prévisionnel, stratégie de commercialisation et rentabilité estimée.
Son ambition : **rendre l’agriculture plus intelligente, plus résiliente et plus rentable au Burkina Faso.**


## Prerequis pour commencer l'exécution du programme:



"Ce qu'il est requis pour commencer avec Notre projet :"

Python 3.8+ - Langage de programmation principal

Ollama - Pour l exécution des modèles de langage localement

Git - Pour le contrôle de version

8GB de RAM minimum - Pour l exécution du modèle Mistral

4GB d espace disque - Pour stocker les modèles et données


### Installation
Les étapes pour installer votre programme :

**1.Cloner le repository**

```{py}

git clone https://github.com/yamsaid/AgroConsulling.git
cd AgroConsulling

```

Créer un environnement virtuel et l'activer

```{py}
python -m venv <nom de l'environnement>
```
```{py}
source <nom de l'environnement>\Scripts\activate
```


**2.Installer les dépendances Python**

```{py}

pip install -r requirements.txt

```


**3.Installer Ollama**

# Sur Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Sur Windows, téléchargez l'installateur 


https://ollama.com

**4.Télécharger le modèle llama3.2:3b**

ollama pull llama3.2:3b

**5.Télécharger le modèle Mistral**

ollama pull mistral

**6.Vérifier l'installation**

```{py}

ollama run mistral "Bonjour, test en français"

```

## Architecture Technique

 Composants Principaux
1. Frontend (Interface Utilisateur)
Framework : Gradio

Localisation : frontend/app.py

Fonctionnalités :

Interface de chat pour conseils agricoles

Affichage des sources documentaires
Design adapté mobile pour agriculteurs


![alt text](<WhatsApp Image 2025-11-05 à 23.30.28_eaf8affd.jpg>)

2. Backend API

Framework : FastAPI

Localisation : src/api.py

Endpoints :

POST /ask - Traitement des questions agricoles

GET /health - Vérification statut système

GET /sources - Liste des documents disponibles



Responsabilités :

Intégration pipeline RAG complet

Gestion des embeddings et recherche vectorielle

Appel au modèle llama3.2:3b pour génération et Mistral comme alternative

3. Moteur RAG (Cœur du Système)

Localisation : src/rag_pipeline.py

Composants :

Embeddings : src/embeddings.py (SentenceTransformers)

Base Vectorielle : src/vector_store.py (Chroma et FAISS en alternative)

LLM : src/llm_handler.py (llama3.2:3b)

Schéma réprsentatif du fonctionnement du chatbot

![alt text](image.png)

4. Gestion des Données Agricoles
Localisation : data/

Fichiers :

corpus.json - Documents techniques agriculture BF

sources.txt - Références des sources


5. Système d'Évaluation

Localisation : evaluation/evaluate.py

Métriques Spécifiques :

Précision Agricole : Exactitude des conseils techniques

Pertinence Contextuelle : Adaptation au contexte burkinabè

Couverture Thématiques : mil, sorgho, maïs, maraîchage

6. Configuration et Déploiement

## Démarrage

**Lancer le serveur uvicorn**
```{py}
uvicorn src.api:app 
```
ou 
```{py}
python -m uvicorn src.api:app
```
**Lancer l'application (l'interface)**
```{py}
python frontend/app.py
```

## Les technologies utilisées

numpy : https://github.com/numpy/numpy/blob/main/LICENSE.txt - library pour le traitement des données

Sentence Transformers : https://github.com/UKPLab/sentence-transformers - Génération d'embeddings multilingues

FAISS : https://github.com/facebookresearch/faiss?tab=MIT-1-ov-file# - Base de données vectorielle

Mistral 7B : https://github.com/ollama/ollama-python?tab=MIT-1-ov-file# - Modèle de langage open source

Gradio : https://github.com/gradio-app/gradio?tab=Apache-2.0-1-ov-file# - Interface utilisateur

FastAPI : https://github.com/fastapi/fastapi?tab=MIT-1-ov-file# - Framework API moderne

Ollama : https://github.com/ollama/ollama-python?tab=MIT-1-ov-file# - Plateforme d'exécution de modèles LLM

Python 3.8+ : https://python.org/ - Langage de programmation principal

uvicorn : https://github.com/Kludex/uvicorn?tab=BSD-3-Clause-1-ov-file


## Contributing

Si vous souhaitez contribuer, lisez le fichier [CONTRIBUTING.md](https://example.org) pour savoir comment le faire.

Règles de contribution :

Respecter les standards de code Python (PEP8)

Ajouter des tests pour les nouvelles fonctionnalités

Documenter toute modification importante

Utiliser des commits descriptifs

## Versions

version : 1.0


## Auteurs

👥 Équipe de Développement

YAMEOGO Saïdou - Data scientist

SANOU Ange Noëlie - Data scientist

NIAMPA Abdoul Fataho - Data scientist

📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](!LICENCE) pour plus d'informations.

