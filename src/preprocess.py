import os
import json
import pdfplumber
import re
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, pdf_folder, output_folder):
        self.pdf_folder = pdf_folder
        self.output_folder = output_folder
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Initialisation des variables pour OCR (importées seulement si nécessaires)
        self.ocr_available = False
        self._init_ocr()
    
    def _init_ocr(self):
        """
        Initialise les outils OCR de manière conditionnelle
        """
        try:
            import pytesseract
            from pdf2image import convert_from_path
            self.ocr_available = True
            logger.info("✓ OCR disponible (pytesseract et pdf2image installés)")
        except ImportError as e:
            logger.warning("⚠ OCR non disponible. Les PDF scannés ne pourront pas être traités.")
            logger.warning("Pour activer l'OCR, installez: pip install pytesseract pdf2image")
            self.ocr_available = False
    
    def extract_text_with_ocr(self, pdf_path):
        """
        Extrait le texte des PDF scannés en utilisant OCR
        """
        if not self.ocr_available:
            return None
            
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            logger.info(f"Tentative OCR sur: {os.path.basename(pdf_path)}")
            
            # Conversion PDF en images
            images = convert_from_path(pdf_path, dpi=300)  # Haute résolution pour meilleure OCR
            
            text = ""
            for i, image in enumerate(images):
                # Configuration Tesseract pour le français
                page_text = pytesseract.image_to_string(image, lang='fra')
                text += f"\n--- Page {i + 1} (OCR) ---\n{page_text}"
                
            return text.strip() if text else None
            
        except Exception as e:
            logger.error(f"Erreur OCR sur {pdf_path}: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extrait le texte d'un PDF avec fallback OCR si nécessaire
        """
        # Première tentative: extraction standard
        try:
            standard_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        standard_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Vérification de la qualité de l'extraction standard
            if self._is_text_quality_good(standard_text):
                return standard_text.strip()
            else:
                logger.warning(f"Texte insuffisant avec extraction standard pour {os.path.basename(pdf_path)}")
                
        except Exception as e:
            logger.warning(f"Échec extraction standard sur {os.path.basename(pdf_path)}: {str(e)}")
        
        # Deuxième tentative: OCR
        if self.ocr_available:
            ocr_text = self.extract_text_with_ocr(pdf_path)
            if ocr_text:
                logger.info(f"✓ OCR réussi pour {os.path.basename(pdf_path)}")
                return ocr_text
            else:
                logger.error(f"Échec OCR sur {os.path.basename(pdf_path)}")
        else:
            logger.error(f"Impossible de traiter le PDF scanné {os.path.basename(pdf_path)} (OCR non disponible)")
        
        return None
    
    def _is_text_quality_good(self, text, min_char_per_page=50, min_pages_with_text=0.5):
        """
        Vérifie si la qualité du texte extrait est suffisante
        """
        if not text:
            return False
        
        # Séparation par pages
        page_sections = re.split(r'--- Page \d+ ---', text)
        pages_with_content = 0
        total_pages = len(page_sections) - 1  # -1 car le split crée un élément vide au début
        
        if total_pages == 0:
            return False
        
        for section in page_sections[1:]:  # Ignorer le premier élément vide
            if len(section.strip()) >= min_char_per_page:
                pages_with_content += 1
        
        # Si au moins 50% des pages ont du contenu, c'est acceptable
        return (pages_with_content / total_pages) >= min_pages_with_text
    
    def clean_text(self, text):
        """
        Nettoie et normalise le texte
        """
        if not text:
            return ""
        
        # Nettoyage de base
        text = re.sub(r'\x00', '', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Normaliser les césures de mots
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Nettoyer les en-têtes/pieds de page
        text = re.sub(r'\bPage\s+\d+\b', '', text)
        text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)
        
        # Nettoyage spécifique pour l'OCR (caractères bizarres)
        text = re.sub(r'[^\w\s.,;:!?()\-–—«»°%€$@/\\+*=]', '', text, flags=re.UNICODE)
        
        return text.strip()
    
    def process_single_pdf(self, pdf_path, filename):
        """
        Traite un seul PDF et retourne ses chunks
        """
        logger.info(f"Traitement de : {filename}")
        
        # Étape 1: Extraction du texte (avec fallback OCR)
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            logger.warning(f"Aucun texte extrait de {filename}")
            return []
        
        # Étape 2: Nettoyage
        cleaned_text = self.clean_text(raw_text)
        
        # Étape 3: Découpage en chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Étape 4: Formatage des résultats
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # Ignorer les chunks trop courts
                processed_chunks.append({
                    "chunk_id": f"{filename}_chunk_{i}",
                    "text": chunk.strip(),
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "extraction_method": "standard" if "OCR" not in raw_text else "ocr"
                })
        
        logger.info(f"✓ {filename} : {len(processed_chunks)} chunks créés")
        return processed_chunks
    
    def process_all_pdfs(self):
        """
        Traite tous les PDFs du dossier
        """
        all_chunks = []
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        
        logger.info(f"Début du traitement de {len(pdf_files)} PDFs...")
        
        # Statistiques
        stats = {
            'total': len(pdf_files),
            'success': 0,
            'failed': 0,
            'ocr_used': 0
        }
        
        for filename in tqdm(pdf_files, desc="Traitement des PDFs"):
            pdf_path = os.path.join(self.pdf_folder, filename)
            chunks = self.process_single_pdf(pdf_path, filename)
            
            if chunks:
                all_chunks.extend(chunks)
                stats['success'] += 1
                # Compter les fichiers traités par OCR
                if any(chunk.get('extraction_method') == 'ocr' for chunk in chunks):
                    stats['ocr_used'] += 1
            else:
                stats['failed'] += 1
        
        # Log des statistiques
        logger.info(f"\n📊 STATISTIQUES D'EXTRACTION:")
        logger.info(f"PDFs traités avec succès: {stats['success']}/{stats['total']}")
        logger.info(f"PDFs échoués: {stats['failed']}/{stats['total']}")
        logger.info(f"PDFs nécessitant OCR: {stats['ocr_used']}/{stats['total']}")
        
        return all_chunks
    
    def save_corpus(self, chunks, output_path):
        """
        Sauvegarde le corpus au format JSON
        """
        # Sauvegarde principale
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # Sauvegarde des sources (pour le livrable sources.txt)
        sources = list(set(chunk['source'] for chunk in chunks))
        sources_path = os.path.join(self.output_folder, 'sources.txt')
        with open(sources_path, 'w', encoding='utf-8') as f:
            for source in sorted(sources):
                f.write(f"{source}\n")
        
        # Statistiques par méthode d'extraction
        ocr_chunks = [chunk for chunk in chunks if chunk.get('extraction_method') == 'ocr']
        standard_chunks = [chunk for chunk in chunks if chunk.get('extraction_method') == 'standard']
        
        logger.info(f"Corpus sauvegardé : {len(chunks)} chunks dans {output_path}")
        logger.info(f"Chunks par extraction standard: {len(standard_chunks)}")
        logger.info(f"Chunks par OCR: {len(ocr_chunks)}")
        logger.info(f"Liste des sources sauvegardée : {len(sources)} fichiers")

def main():
    # Configuration des chemins
    PDF_FOLDER = "./data/raw_pdfs"
    OUTPUT_FOLDER = "./data/processed"
    CORPUS_PATH = "./data/corpus.json"
    
    # Création des dossiers si nécessaire
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Vérification que le dossier des PDFs existe
    if not os.path.exists(PDF_FOLDER):
        logger.error(f"Le dossier {PDF_FOLDER} n'existe pas!")
        return
    
    # Traitement
    processor = PDFProcessor(PDF_FOLDER, OUTPUT_FOLDER)
    all_chunks = processor.process_all_pdfs()
    
    # Sauvegarde
    if all_chunks:
        processor.save_corpus(all_chunks, CORPUS_PATH)
        
        # Statistiques finales
        total_chunks = len(all_chunks)
        total_files = len(set(chunk['source'] for chunk in all_chunks))
        avg_chunks_per_file = total_chunks / total_files if total_files > 0 else 0
        
        logger.info("\n" + "="*50)
        logger.info("STATISTIQUES FINALES:")
        logger.info(f"PDFs traités: {total_files}")
        logger.info(f"Chunks créés: {total_chunks}")
        logger.info(f"Moyenne de chunks par PDF: {avg_chunks_per_file:.1f}")
        logger.info("="*50)
    else:
        logger.error("Aucun chunk n'a été créé!")

if __name__ == "__main__":
    main()