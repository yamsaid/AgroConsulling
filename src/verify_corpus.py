import json
import random

def verify_corpus(corpus_path, num_samples=5):
    """
    Vérifie la qualité du corpus généré
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    print(f"Corpus chargé : {len(corpus)} chunks")
    
    # Échantillons aléatoires
    samples = random.sample(corpus, min(num_samples, len(corpus)))
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"ÉCHANTILLON {i+1}:")
        print(f"Source: {sample['source']}")
        print(f"Chunk ID: {sample['chunk_id']}")
        print(f"Taille: {len(sample['text'])} caractères")
        print(f"{'-'*40}")
        print(f"TEXTE:\n{sample['text'][:500]}...")  # Premier 500 caractères
        print(f"{'='*60}")

if __name__ == "__main__":
    verify_corpus("./data/corpus.json")