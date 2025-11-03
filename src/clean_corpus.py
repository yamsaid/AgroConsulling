import json
import re
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def advanced_clean_text(text):
    """
    Version avancée du nettoyage pour le post-traitement
    """
    if not text:
        return ""
    
    # Nettoyage en plusieurs passes
    cleaning_steps = [
        # 1. Nettoyage des artefacts d'extraction
        (r'---\s*Page\s*\d+\s*---', ''),
        (r'---+\s*--+', ''),
        (r'_{3,}', ''),
        (r'\*{3,}', ''),
        
        # 2. Nettoyage des séparateurs de ligne
        (r'\n+', ' '),
        (r'\r+', ' '),
        (r'\t+', ' '),
        
        # 3. Correction des césures
        (r'(\w+)-\s+(\w+)', r'\1\2'),
        
        # 4. Nettoyage des caractères indésirables
        (r'[^\w\s.,;:!?()\-–—«»°%€$@/\\+*=&]', ''),
        
        # 5. Nettoyage des numéros de page
        (r'\bPage\s+\d+\b', ''),
        (r'\b\d+\s*/\s*\d+\b', ''),
        (r'\bvol\.?\s*\d+\b', '', re.IGNORECASE),
        (r'\bno\.?\s*\d+\b', '', re.IGNORECASE),
        
        # 6. Normalisation des espaces
        (r'\s+', ' '),
        
        # 7. Correction ponctuation
        (r'\s+([.,;:!?)])', r'\1'),
        (r'([(])\s+', r'\1'),
        (r'\s+–\s+', ' – '),  # Garde les tirets cadratins espacés
        (r'\s+-\s+', ' - '),  # Garde les traits d'union espacés
    ]
    
    cleaned_text = text
    for step in cleaning_steps:
        if len(step) == 3:
            # Tuple avec flags: (pattern, replacement, flags)
            pattern, replacement, flags = step
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=flags)
        elif len(step) == 2:
            # Tuple sans flags: (pattern, replacement)
            pattern, replacement = step
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        else:
            logger.warning(f"Étape de nettoyage invalide ignorée: {step}")
    
    return cleaned_text.strip()

def clean_existing_corpus(input_path, output_path):
    """
    Nettoie un corpus existant sans reprocesser les PDFs
    """
    logger.info(f"Chargement du corpus depuis {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    logger.info(f"Corpus chargé: {len(corpus)} chunks")
    
    # Nettoyage de chaque chunk
    cleaned_corpus = []
    removed_chunks = 0
    
    for chunk in tqdm(corpus, desc="Nettoyage des chunks"):
        original_text = chunk['text']
        cleaned_text = advanced_clean_text(original_text)
        
        # Ne garder que les chunks qui ont encore du contenu significatif
        if len(cleaned_text) >= 15:  # Au moins 15 caractères
            chunk['text'] = cleaned_text
            #chunk['original_length'] = len(original_text)
            chunk['length'] = len(cleaned_text)
            cleaned_corpus.append(chunk)
        else:
            removed_chunks += 1
    
    # Sauvegarde du corpus nettoyé
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_corpus, f, ensure_ascii=False, indent=2)
    
    # Statistiques
    logger.info(f"\n📊 RÉSULTATS DU NETTOYAGE:")
    logger.info(f"Chunks initiaux: {len(corpus)}")
    logger.info(f"Chunks après nettoyage: {len(cleaned_corpus)}")
    logger.info(f"Chunks supprimés (trop courts): {removed_chunks}")
    logger.info(f"Taux de conservation: {(len(cleaned_corpus)/len(corpus))*100:.1f}%")
    
    # Aperçu des améliorations
    if len(cleaned_corpus) > 0:
        logger.info("\n🔍 APERÇU AVANT/APRÈS:")
        for i in range(min(3, len(corpus))):
            original = corpus[i]['text'][:100] + "..." if len(corpus[i]['text']) > 100 else corpus[i]['text']
            cleaned = cleaned_corpus[i]['text'][:100] + "..." if len(cleaned_corpus[i]['text']) > 100 else cleaned_corpus[i]['text']
            logger.info(f"Chunk {i+1}:")
            logger.info(f"  AVANT: {original}")
            logger.info(f"  APRÈS: {cleaned}")
            logger.info(f"  " + "-"*50)

def main():
    input_corpus = "./data/corpus.json"
    output_corpus = "./data/corpus_cleaned.json"
    
    clean_existing_corpus(input_corpus, output_corpus)
    logger.info(f"✓ Corpus nettoyé sauvegardé dans: {output_corpus}")

if __name__ == "__main__":
    main()