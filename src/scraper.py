import os
import json
import csv
import pdfplumber
import re
import requests
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import time
import urllib.robotparser
from urllib.parse import urljoin
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
import io
import clean_corpus
import embeddings


# Configuration du logging sans emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Structure pour les résultats de scraping"""
    url: str
    source_id: str
    status: str
    chunks_count: int
    error_message: Optional[str] = None
    content_type: Optional[str] = None
    bytes_downloaded: int = 0
    processing_time: float = 0.0
    source_institution: Optional[str] = None

class FrenchSmartSplitter:
    """Splitter intelligent qui respecte les phrases françaises avec NLTK"""
    
    def __init__(self, chunk_size=450, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Configure NLTK avec gestion des erreurs"""
        try:
            # Vérifier punkt_tab (nouveau format NLTK pour français)
            try:
                nltk.data.find('tokenizers/punkt_tab')
                logger.info("Modèles NLTK punkt_tab déjà installés")
            except LookupError:
                # Vérifier punkt (ancien format)
                try:
                    nltk.data.find('tokenizers/punkt')
                    logger.info("Modèles NLTK punkt déjà installés")
                except LookupError:
                    raise LookupError("Aucun modèle NLTK trouvé")
        except LookupError:
            logger.info("Téléchargement des modèles NLTK...")
            try:
                # Contournement SSL pour le téléchargement
                import ssl
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                
                # Télécharger punkt_tab (nouveau format, supporte le français)
                try:
                    nltk.download('punkt_tab', quiet=True)
                    logger.info("Modèle NLTK punkt_tab téléchargé avec succès!")
                except Exception as e1:
                    logger.warning(f"Échec téléchargement punkt_tab: {e1}, essai avec punkt...")
                    # Fallback vers punkt (ancien format)
                    nltk.download('punkt', quiet=True)
                    logger.info("Modèle NLTK punkt téléchargé avec succès!")
                    
            except Exception as e:
                logger.error(f"Erreur téléchargement NLTK: {e}")
                logger.warning("Le tokenisation française pourrait ne pas fonctionner correctement")
    
    def split_text_respectueux(self, text):
        """
        Découpe le texte en respectant les phrases françaises
        """
        from nltk.tokenize import sent_tokenize
        
        if not text or len(text.strip()) == 0:
            return []
        
        # Tokenisation en phrases françaises
        try:
            # Essayer avec punkt_tab (nouveau format)
            try:
                phrases = sent_tokenize(text, language='french')
            except (LookupError, OSError) as e:
                # Si punkt_tab n'est pas disponible, essayer sans spécifier la langue
                logger.warning(f"Tokenisation française échouée, essai sans langue spécifique: {e}")
                try:
                    phrases = sent_tokenize(text)
                except Exception as e2:
                    logger.warning(f"Erreur tokenisation NLTK, utilisation fallback: {e2}")
                    # Fallback: séparation par points
                    phrases = [p.strip() for p in text.split('.') if p.strip()]
                    phrases = [p + '.' for p in phrases if not p.endswith('.')]
        except Exception as e:
            logger.warning(f"Erreur tokenisation NLTK, utilisation fallback: {e}")
            # Fallback: séparation par points
            phrases = [p.strip() for p in text.split('.') if p.strip()]
            phrases = [p + '.' for p in phrases if not p.endswith('.')]
        
        chunks = []
        chunk_actuel = ""
        longueur_actuelle = 0
        
        for phrase in phrases:
            # Nettoyer la phrase
            phrase = phrase.strip()
            if not phrase:
                continue
                
            longueur_phrase = len(phrase)
            
            # Vérifier si l'ajout dépasse la limite
            if longueur_actuelle + longueur_phrase > self.chunk_size and chunk_actuel:
                # Sauvegarder le chunk actuel
                chunks.append(chunk_actuel.strip())
                
                # Préparer l'overlap pour le chunk suivant
                if self.overlap > 0:
                    phrases_overlap = self._obtenir_phrases_overlap(chunk_actuel)
                    chunk_actuel = phrases_overlap
                    longueur_actuelle = len(chunk_actuel)
                else:
                    chunk_actuel = ""
                    longueur_actuelle = 0
            
            # Ajouter la phrase actuelle
            if chunk_actuel:
                # Ajouter un espace entre les phrases
                chunk_actuel += " " + phrase
                longueur_actuelle += longueur_phrase + 1
            else:
                chunk_actuel = phrase
                longueur_actuelle = longueur_phrase
        
        # Ajouter le dernier chunk
        if chunk_actuel and chunk_actuel.strip():
            chunks.append(chunk_actuel.strip())
        
        return chunks
    
    def _obtenir_phrases_overlap(self, texte):
        """Extrait les dernières phrases pour l'overlap"""
        from nltk.tokenize import sent_tokenize
        
        try:
            # Essayer avec langue française
            try:
                phrases = sent_tokenize(texte, language='french')
            except (LookupError, OSError):
                # Essayer sans langue spécifique
                try:
                    phrases = sent_tokenize(texte)
                except Exception:
                    # Fallback
                    phrases = [p.strip() for p in texte.split('.') if p.strip()]
                    phrases = [p + '.' for p in phrases if not p.endswith('.')]
        except Exception:
            # Fallback
            phrases = [p.strip() for p in texte.split('.') if p.strip()]
            phrases = [p + '.' for p in phrases if not p.endswith('.')]
        
        texte_overlap = ""
        
        # Prendre les phrases depuis la fin jusqu'à atteindre l'overlap
        for phrase in reversed(phrases):
            phrase = phrase.strip()
            if not phrase:
                continue
                
            if len(texte_overlap) + len(phrase) <= self.overlap:
                texte_overlap = phrase + " " + texte_overlap if texte_overlap else phrase
            else:
                break
        
        return texte_overlap.strip()

class StrictEthicalWebScraper:
    """Scraper qui respecte STRICTEMENT robots.txt - Version hackathon"""
    
    def __init__(self, delay_between_requests: float = 5.0, max_retries: int = 2):
        self.delay = delay_between_requests
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Academic-Research-Bot-Hackathon/1.0 (+https://github.com/our-project) for educational non-commercial research',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'From': 'research@education.org',
        })
        self.robot_parsers = {}
        self.blocked_domains = set()
        
    def can_fetch(self, url: str) -> bool:
        """
        Respect STRICT du robots.txt - Aucun contournement
        """
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            if base_url in self.blocked_domains:
                return False
            
            if base_url not in self.robot_parsers:
                rp = urllib.robotparser.RobotFileParser()
                robots_url = urljoin(base_url, '/robots.txt')
                try:
                    rp.set_url(robots_url)
                    rp.read()
                    self.robot_parsers[base_url] = rp
                    logger.info(f"robots.txt chargé pour {base_url}")
                    
                except Exception as e:
                    logger.warning(f"Impossible de charger robots.txt pour {base_url}: {e}")
                    # SI robots.txt inaccessible, on assume le pire cas
                    self.robot_parsers[base_url] = None
                    return False  # Blocage par défaut si inaccessible
            
            if self.robot_parsers[base_url] is None:
                return False  # Blocage si robots.txt inaccessible
                
            can_access = self.robot_parsers[base_url].can_fetch(
                self.session.headers['User-Agent'], url
            )
            
            if not can_access:
                logger.warning(f"ACCES REFUSE par robots.txt: {url}")
                self.blocked_domains.add(base_url)
                
            return can_access
            
        except Exception as e:
            logger.error(f"Erreur vérification robots.txt: {e}")
            return False  # Blocage en cas d'erreur

    def get_crawl_delay(self, url: str) -> float:
        """Récupère le délai de crawl recommandé depuis robots.txt"""
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            if base_url in self.robot_parsers and self.robot_parsers[base_url]:
                delay = self.robot_parsers[base_url].crawl_delay(
                    self.session.headers['User-Agent']
                )
                return delay if delay else self.delay
        except:
            pass
        return self.delay

    def respectful_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """Requête ultra-respectueuse pour hackathon"""
        
        # Vérification STRICTE robots.txt
        if not self.can_fetch(url):
            return None
        
        # Délai conservateur
        crawl_delay = max(self.get_crawl_delay(url), 5.0)  # Minimum 5 secondes
        
        for attempt in range(self.max_retries):
            try:
                time.sleep(crawl_delay)
                
                response = self.session.request(
                    method, url, 
                    timeout=30,
                    allow_redirects=True,
                    **kwargs
                )
                
                # Respect des codes HTTP
                if response.status_code == 403:
                    logger.warning(f"Accès interdit (403) - Arrêt immédiat: {url}")
                    self.blocked_domains.add(urlparse(url).netloc)
                    return None
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 120))  # 2 minutes
                    logger.warning(f"Rate limit détecté - Pause de {retry_after}s: {url}")
                    time.sleep(retry_after)
                    continue
                
                if response.status_code == 503:
                    logger.warning(f"Service indisponible (503) - Pause longue: {url}")
                    time.sleep(60)  # Pause d'1 minute
                    continue
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [403, 404, 500, 503]:
                    logger.warning(f"Erreur HTTP {e.response.status_code} - Arrêt: {url}")
                    return None
                logger.error(f"Erreur HTTP pour {url}: {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.delay * (attempt + 1))
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout pour {url}, tentative {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.delay * (attempt + 1))
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Erreur requête {url} (tentative {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.delay * (attempt + 1))
                
        return None

class PDFProcessor:
    """Traitement robuste des PDFs"""
    
    def __init__(self, output_folder: str):
        self.output_folder = Path(output_folder)
        # Utilisation du splitter intelligent pour les PDFs aussi
        self.smart_splitter = FrenchSmartSplitter(chunk_size=450, overlap=50)
        # Fallback avec LangChain
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Nettoyage avancé du texte"""
        if not text:
            return ""
        
        # Suppression caractères nuls et contrôle
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalisation des espaces
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Césures de mots
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Suppression patterns non désirés
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = re.sub(r'\{[^\}]*\}', '', text)
        
        return text.strip()
    
    def process_single_pdf(self, pdf_path: str, source_id: str, source_institution: str) -> List[Dict]:
        """Traitement d'un PDF unique avec institution source"""
        chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            cleaned = self.clean_text(page_text)
                            if cleaned:
                                full_text.append(cleaned)
                    except Exception as e:
                        logger.warning(f"Erreur page {page_num} de {source_id}: {e}")
                
                if full_text:
                    complete_text = "\n\n".join(full_text)
                    
                    # ESSAI DU CHUNKING INTELLIGENT D'ABORD
                    try:
                        text_chunks = self.smart_splitter.split_text_respectueux(complete_text)
                        extraction_method = "smart_pdf"
                    except Exception as e:
                        logger.warning(f"Chunking intelligent échoué, fallback: {e}")
                        text_chunks = self.text_splitter.split_text(complete_text)
                        extraction_method = "fallback_pdf"
                    
                    for i, chunk in enumerate(text_chunks):
                        if len(chunk.strip()) > 50:
                            chunks.append({
                                "chunk_id": f"{source_id}_pdf_chunk_{i}",
                                "text": chunk.strip(),
                                "source": os.path.basename(pdf_path),
                                "source_institution": source_institution,
                                "chunk_index": i,
                                "total_chunks": len(text_chunks),
                                "extraction_method": extraction_method,
                                "char_length": len(chunk),
                                "word_count": len(chunk.split())
                            })
                    
                    logger.info(f"PDF {source_id}: {len(chunks)} chunks crees ({extraction_method})")
                else:
                    logger.warning(f"Aucun texte extrait du PDF {source_id}")
                    
        except Exception as e:
            logger.error(f"Erreur traitement PDF {source_id}: {e}")
        
        return chunks

class WebContentProcessor:
    """Processeur de contenu web éthique et robuste"""
    
    def __init__(self, output_folder: str, csv_log_path: str = None):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        self.temp_folder = self.output_folder / "temp_downloads"
        self.temp_folder.mkdir(exist_ok=True)
        
        self.csv_log_path = csv_log_path or self.output_folder / "scraping_log.csv"
        
        # Splitter intelligent NLTK pour le web
        self.smart_splitter = FrenchSmartSplitter(chunk_size=450, overlap=50)
        
        # Fallback avec LangChain
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Utilisation du scraper éthique strict
        self.scraper = StrictEthicalWebScraper(delay_between_requests=5.0)
        self.pdf_processor = PDFProcessor(output_folder)
        self.results: List[ScrapingResult] = []
        
        # Fichier pour enregistrer les sources
        self.source_file = self.output_folder / "sources.txt"
        # Fichier pour les URLs bloquées
        self.blocked_file = self.output_folder / "blocked_urls_report.json"
        
        self._init_csv_log()
        self._init_source_file()
    
    def _init_csv_log(self):
        """Initialise le fichier CSV de log"""
        if not self.csv_log_path.exists():
            with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'url', 'source_id', 'status', 'chunks_count',
                    'content_type', 'bytes_downloaded', 'processing_time', 'error_message', 'source_institution'
                ])
                writer.writeheader()
    
    def _init_source_file(self):
        """Initialise le fichier sources.txt"""
        if not self.source_file.exists():
            with open(self.source_file, 'w', encoding='utf-8') as f:
                f.write("# Fichier des sources traitées avec succès\n")
                f.write("# Format: URL | Source/Institution | Date de traitement\n")
                f.write("=" * 80 + "\n")
    
    def _log_to_csv(self, result: ScrapingResult):
        """Enregistre un résultat dans le CSV"""
        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'url', 'source_id', 'status', 'chunks_count',
                'content_type', 'bytes_downloaded', 'processing_time', 'error_message', 'source_institution'
            ])
            row = asdict(result)
            row['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow(row)
    
    def _log_source_to_file(self, url: str, source_institution: str, status: str = "success"):
        """Enregistre la source dans le fichier sources.txt"""
        if status == "success":
            with open(self.source_file, 'a', encoding='utf-8') as f:
                f.write(f"{url} | {source_institution} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def download_content_with_auto_detect(self, url: str, source_id: str) -> Optional[Tuple[str, str]]:
        """
        Télécharge le contenu avec détection automatique du type
        Retourne (chemin_fichier, type_contenu) ou None
        """
        response = self.scraper.respectful_request(url)
        if not response:
            return None
        
        content_type = response.headers.get('content-type', '').lower()
        content_length = len(response.content)
        
        # Vérifier si le contenu est trop petit
        if content_length < 100:
            logger.warning(f"Contenu trop petit: {url} ({content_length} octets)")
            return None
        
        parsed_url = urlparse(url)
        filename = f"{source_id}_{hashlib.md5(url.encode()).hexdigest()[:8]}"
        
        # DÉTECTION PDF AVANCÉE
        is_pdf = (
            'pdf' in content_type or 
            url.lower().endswith('.pdf') or
            response.content.startswith(b'%PDF') or
            '.pdf' in response.headers.get('content-disposition', '').lower() or
            # Règles spécifiques pour les APIs connues
            any(domain in url for domain in [
                'openknowledge.fao.org',
                'cgspace.cgiar.org', 
                'agritrop.cirad.fr',
                'hal.archives-ouvertes.fr'
            ]) and content_length > 50000  # Fichiers > 50KB sur ces domaines
        )
        
        if is_pdf:
            file_path = self.temp_folder / f"{filename}.pdf"
            try:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # Validation PDF
                try:
                    with pdfplumber.open(file_path) as test_pdf:
                        if len(test_pdf.pages) > 0:
                            logger.info(f"PDF valide detecte: {url} ({len(test_pdf.pages)} pages)")
                            return str(file_path), "pdf"
                        else:
                            logger.warning(f"PDF invalide (0 pages): {url}")
                            os.remove(file_path)
                            return None
                except Exception as e:
                    logger.warning(f"Fichier non-PDF: {url} - {e}")
                    os.remove(file_path)
                    return None
                    
            except Exception as e:
                logger.error(f"Erreur sauvegarde PDF {url}: {e}")
                return None
        
        # DÉTECTION HTML/TEXTE
        elif 'html' in content_type or 'text' in content_type or content_length < 1000000:
            return None  # Laissé à extract_text_from_webpage
        
        # TYPE INCONNU - Essayer de détecter
        else:
            logger.warning(f"Type de contenu inconnu: {url} - Content-Type: {content_type}")
            return None

    def extract_text_from_webpage(self, url: str) -> Optional[str]:
        """Extraction éthique de texte web"""
        response = self.scraper.respectful_request(url)
        if not response:
            return None
        
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type and 'text' not in content_type:
            logger.warning(f"Contenu non-HTML pour {url}: {content_type}")
            return None
        
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Suppression éléments non désirés
            for element in soup(["script", "style", "nav", "header", "footer", 
                                "aside", "form", "noscript", "iframe"]):
                element.decompose()
            
            main_content = self._extract_main_content(soup)
            text = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
            
            # Nettoyage
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = '\n'.join(lines)
            
            if len(clean_text) > 200:
                logger.info(f"Texte web extrait: {len(clean_text)} caracteres")
                return clean_text
            else:
                logger.warning(f"Texte trop court: {len(clean_text)} caracteres")
                return None
                
        except Exception as e:
            logger.error(f"Erreur extraction web {url}: {e}")
            return None

    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Identification du contenu principal"""
        selectors = [
            'article', 'main', '[role="main"]',
            '.content', '.main-content', '.post-content',
            '.entry-content', '#content', '.article-body'
        ]
        
        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                return content
        
        return soup.find('body')

    def clean_text(self, text: str) -> str:
        """Nettoyage avancé du texte"""
        if not text:
            return ""
        
        # Suppression caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalisation espaces
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Césures
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Nettoyage spécifique web
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        return text.strip()

    def smart_chunk_text(self, text: str, source_id: str, source_institution: str, url: str) -> List[Dict]:
        """Découpe intelligent le texte en respectant les phrases avec NLTK"""
        chunks = []
        
        try:
            # ESSAI DU CHUNKING INTELLIGENT D'ABORD
            text_chunks = self.smart_splitter.split_text_respectueux(text)
            extraction_method = "smart_web"
            
        except Exception as e:
            logger.warning(f"Chunking intelligent échoué pour {source_id}, fallback: {e}")
            # FALLBACK VERS L'ANCIENNE METHODE
            text_chunks = self.text_splitter.split_text(text)
            extraction_method = "fallback_web"
        
        for i, chunk in enumerate(text_chunks):
            if len(chunk.strip()) > 50:  # Éviter les chunks trop courts
                chunks.append({
                    "chunk_id": f"{source_id}_chunk_{i}",
                    "text": chunk.strip(),
                    "source": url,
                    "source_url": url,
                    "source_institution": source_institution,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "extraction_method": extraction_method,
                    "char_length": len(chunk),
                    "word_count": len(chunk.split())
                })
        
        logger.info(f"Chunking {source_id}: {len(chunks)} chunks crees ({extraction_method})")
        return chunks

    def fallback_download_and_analyze(self, url: str, source_id: str, source_institution: str) -> List[Dict]:
        """Fallback ultime pour le contenu problématique"""
        response = self.scraper.respectful_request(url)
        if not response or len(response.content) < 100:
            return []
        
        content = response.content
        chunks = []
        
        # Essayer PDF même sans signature
        if len(content) > 1000:  # Fichier assez grand
            temp_path = self.temp_folder / f"fallback_{source_id}.bin"
            
            try:
                # Sauvegarder et tester comme PDF
                with open(temp_path, 'wb') as f:
                    f.write(content)
                
                # Essayer d'ouvrir avec pdfplumber
                try:
                    with pdfplumber.open(temp_path) as pdf:
                        full_text = []
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                full_text.append(self.clean_text(page_text))
                        
                        if full_text:
                            complete_text = "\n\n".join(full_text)
                            text_chunks = self.smart_splitter.split_text_respectueux(complete_text)
                            
                            for i, chunk in enumerate(text_chunks):
                                if len(chunk.strip()) > 50:
                                    chunks.append({
                                        "chunk_id": f"{source_id}_fallback_pdf_chunk_{i}",
                                        "text": chunk.strip(),
                                        "source": url,
                                        "source_url": url,
                                        "source_institution": source_institution,
                                        "chunk_index": i,
                                        "total_chunks": len(text_chunks),
                                        "extraction_method": "fallback_pdf_detection",
                                        "char_length": len(chunk)
                                    })
                            
                            logger.info(f"PDF detecte en fallback: {url} ({len(chunks)} chunks)")
                except:
                    pass  # Ce n'est pas un PDF
                    
                finally:
                    # Nettoyage
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Erreur analyse fallback {url}: {e}")
        
        return chunks

    def process_url(self, url: str, source_id: str, source_institution: str) -> Tuple[List[Dict], ScrapingResult]:
        """Traitement complet d'une URL avec détection de contenu avancée"""
        start_time = time.time()
        
        logger.info(f"Traitement de: {url} (Source: {source_institution})")
        
        result = ScrapingResult(
            url=url,
            source_id=source_id,
            status="failed",
            chunks_count=0,
            source_institution=source_institution
        )
        
        if not self.scraper.can_fetch(url):
            result.error_message = "Bloque par robots.txt"
            result.status = "blocked"
            self._log_to_csv(result)
            return [], result
        
        chunks = []
        
        # ÉTAPE 1: Tentative de téléchargement avec détection automatique
        downloaded_content = self.download_content_with_auto_detect(url, source_id)
        
        if downloaded_content:
            file_path, content_type = downloaded_content
            if content_type == "pdf":
                chunks = self.pdf_processor.process_single_pdf(file_path, source_id, source_institution)
                result.content_type = "pdf"
                
                # Nettoyage du fichier temporaire
                try:
                    os.remove(file_path)
                except:
                    pass
        
        # ÉTAPE 2: Fallback - Extraction web standard
        if not chunks:
            web_text = self.extract_text_from_webpage(url)
            if web_text:
                cleaned_text = self.clean_text(web_text)
                chunks = self.smart_chunk_text(cleaned_text, source_id, source_institution, url)
                result.content_type = "html"
        
        # ÉTAPE 3: Fallback ultime - Téléchargement direct pour analyse
        if not chunks:
            chunks = self.fallback_download_and_analyze(url, source_id, source_institution)
            if chunks:
                result.content_type = "binary_analyzed"
        
        # Mise à jour résultat
        if chunks:
            for chunk in chunks:
                chunk['source_url'] = url
            
            result.status = "success"
            result.chunks_count = len(chunks)
            self._log_source_to_file(url, source_institution, "success")
            logger.info(f"{source_id}: {len(chunks)} chunks crees (methode: {result.content_type})")
        else:
            result.error_message = "Aucun contenu extractible"
            self._log_source_to_file(url, source_institution, "failed")
            logger.warning(f"Echec extraction: {url}")
        
        result.processing_time = time.time() - start_time
        self._log_to_csv(result)
        self.results.append(result)
        
        return chunks, result

    def load_sources_from_csv(self, csv_path: str) -> Dict[str, str]:
        """Charge les sources depuis le CSV historique"""
        sources_map = {}
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Détection du délimiteur
                sample = f.read(2048)
                f.seek(0)
                
                # Essayer différents délimiteurs
                for delimiter in [';', ',', '\t']:
                    try:
                        reader = csv.DictReader(f, delimiter=delimiter)
                        if 'url' in reader.fieldnames and 'source' in reader.fieldnames:
                            for row in reader:
                                if row['url'] and row['source']:
                                    sources_map[row['url']] = row['source']
                            logger.info(f"Sources chargees depuis CSV avec delimiteur '{delimiter}': {len(sources_map)} entrees")
                            return sources_map
                        f.seek(0)
                    except:
                        f.seek(0)
                        continue
                
                logger.error("Impossible de determiner le format du CSV")
                
        except Exception as e:
            logger.error(f"Erreur chargement sources CSV {csv_path}: {e}")
        
        return sources_map

    def handle_blocked_urls(self, urls_to_process: List[Tuple[str, str, str]]) -> Dict:
        """
        Analyse les URLs et identifie celles bloquées par robots.txt
        Retourne un rapport détaillé
        """
        blocked_report = {
            'total_urls': len(urls_to_process),
            'blocked_urls': [],
            'accessible_urls': [],
            'blocked_domains': set()
        }
        
        logger.info("Analyse préliminaire du robots.txt...")
        
        for url, source_id, source_institution in tqdm(urls_to_process, desc="Vérification robots.txt"):
            if self.scraper.can_fetch(url):
                blocked_report['accessible_urls'].append((url, source_id, source_institution))
            else:
                blocked_report['blocked_urls'].append((url, source_id, source_institution))
                domain = urlparse(url).netloc
                blocked_report['blocked_domains'].add(domain)
        
        # Rapport détaillé
        logger.info("\n" + "="*60)
        logger.info("RAPPORT ROBOTS.TXT")
        logger.info("="*60)
        logger.info(f"URLs totales: {blocked_report['total_urls']}")
        logger.info(f"URLs accessibles: {len(blocked_report['accessible_urls'])}")
        logger.info(f"URLs bloquées: {len(blocked_report['blocked_urls'])}")
        logger.info(f"Domaines bloqués: {len(blocked_report['blocked_domains'])}")
        
        for domain in sorted(blocked_report['blocked_domains']):
            count = sum(1 for url, _, _ in blocked_report['blocked_urls'] if domain in url)
            logger.info(f"  - {domain}: {count} URLs bloquées")
        
        return blocked_report

    def process_urls_from_csv(self, csv_path: str, sources_csv_path: str, 
                              url_column: str = 'url', id_column: str = 'id') -> List[Dict]:
        """Traitement d'URLs depuis un CSV avec mapping des sources"""
        urls_to_process = []
        sources_map = self.load_sources_from_csv(sources_csv_path)
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                sample = f.read(2048)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                except Exception:
                    dialect = csv.excel
                reader = csv.DictReader(f, dialect=dialect)
                for row in reader:
                    if url_column in row and id_column in row:
                        url = row[url_column]
                        source_id = row[id_column]
                        # Récupération de la source depuis le mapping
                        source_institution = sources_map.get(url, "Source inconnue")
                        urls_to_process.append((url, source_id, source_institution))
                    else:
                        logger.error(f"Colonnes manquantes: {url_column}, {id_column}")
                        break
        except Exception as e:
            logger.error(f"Erreur lecture CSV {csv_path}: {e}")
            return []
        
        logger.info(f"{len(urls_to_process)} URLs chargees depuis CSV avec sources")
        
        # ANALYSE PRELIMINAIRE DES BLOQUAGES
        blocked_report = self.handle_blocked_urls(urls_to_process)
        
        # SAUVEGARDE DU RAPPORT DES BLOQUAGES
        with open(self.blocked_file, 'w', encoding='utf-8') as f:
            json.dump({
                'blocked_urls': blocked_report['blocked_urls'],
                'blocked_domains': list(blocked_report['blocked_domains']),
                'accessible_urls_count': len(blocked_report['accessible_urls']),
                'blocked_urls_count': len(blocked_report['blocked_urls']),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        
        # TRAITEMENT SEULEMENT DES URLs AUTORISEES
        return self.process_urls(blocked_report['accessible_urls'])

    def process_urls(self, url_list: List[Tuple[str, str, str]]) -> List[Dict]:
        """Traitement de liste d'URLs avec sources"""
        all_chunks = []
        
        logger.info(f"Debut traitement de {len(url_list)} URLs...")
        
        for url, source_id, source_institution in tqdm(url_list, desc="Traitement URLs"):
            chunks, result = self.process_url(url, source_id, source_institution)
            if chunks:
                all_chunks.extend(chunks)
        
        self._print_summary()
        return all_chunks

    def _print_summary(self):
        """Affichage du rapport final"""
        total = len(self.results)
        success = sum(1 for r in self.results if r.status == "success")
        blocked = sum(1 for r in self.results if r.status == "blocked")
        failed = sum(1 for r in self.results if r.status == "failed")
        
        total_chunks = sum(r.chunks_count for r in self.results)
        
        # Statistiques des méthodes de chunking
        smart_chunks = sum(1 for r in self.results if r.status == "success" and r.chunks_count > 0)
        
        logger.info("\n" + "="*60)
        logger.info("RAPPORT DE SCRAPING AVEC NLTK")
        logger.info("="*60)
        logger.info(f"Total URLs traitees: {total}")
        logger.info(f"Succes: {success}")
        logger.info(f"Bloquees (robots.txt): {blocked}")
        logger.info(f"Echecs: {failed}")
        logger.info(f"Total chunks crees: {total_chunks}")
        logger.info(f"Chunking intelligent: {smart_chunks} URLs")
        logger.info(f"Log CSV: {self.csv_log_path}")
        logger.info(f"Fichier sources: {self.source_file}")
        logger.info(f"Rapport blocages: {self.blocked_file}")
        logger.info("="*60)

def main():
    """Fonction principale"""
    # Configuration
    OUTPUT_FOLDER = "./data/processed"
    CSV_INPUT = "./data/sources.csv"  # Fichier CSV avec URLs à traiter
    SOURCES_CSV = "./data/sources.csv"  # Fichier CSV avec mapping des sources
    CORPUS_OUTPUT = "./data/corpus_dearty.json"
    
    # Création dossiers
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Initialisation processeur
    processor = WebContentProcessor(OUTPUT_FOLDER)
    
    # Option 1: Traitement depuis CSV avec mapping des sources
    if os.path.exists(CSV_INPUT) and os.path.exists(SOURCES_CSV):
        logger.info(f"Chargement URLs depuis CSV: {CSV_INPUT}")
        logger.info(f"Chargement sources depuis: {SOURCES_CSV}")
        all_chunks = processor.process_urls_from_csv(
            CSV_INPUT,
            SOURCES_CSV,
            url_column='url',
            id_column='id'
        )
    else:
        # Option 2: Liste manuelle avec sources
        logger.info("Fichiers CSV non trouves, utilisation liste manuelle")
        urls_to_process = [
            ("https://revuesciences-techniquesburkina.org/index.php/sciences_naturelles_et_appliquee/article/view/608/441", "revue_agriculture_1", "Sciences Naturelles et Appliquées"),
            ("https://faolex.fao.org/docs/pdf/bkf198258.pdf", "fao_1", "FAO"),
        ]
        all_chunks = processor.process_urls(urls_to_process)
    
    # Sauvegarde corpus avec sources
    if all_chunks:
        with open(CORPUS_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Corpus sauvegarde: {CORPUS_OUTPUT} ({len(all_chunks)} chunks)")
        
        # Rapport des sources utilisées
        sources_used = set(chunk.get('source_institution', 'Inconnue') for chunk in all_chunks)
        logger.info(f"Sources utilisees dans le corpus: {len(sources_used)}")
        for source in sorted(sources_used):
            count = sum(1 for chunk in all_chunks if chunk.get('source_institution') == source)
            logger.info(f"  - {source}: {count} chunks")
            
        # Rapport des méthodes d'extraction
        methods = set(chunk.get('extraction_method', 'Inconnu') for chunk in all_chunks)
        logger.info(f"Methodes d'extraction utilisees: {len(methods)}")
        for method in sorted(methods):
            count = sum(1 for chunk in all_chunks if chunk.get('extraction_method') == method)
            logger.info(f"  - {method}: {count} chunks")
    else:
        logger.warning("Aucun chunk cree, corpus non sauvegarde")
    
    #nettoyage
    input_corpus = "./data/corpus_dearty.json"
    output_corpus = "./data/corpus.json"
    
    clean_corpus.clean_existing_corpus(input_corpus, output_corpus)
    logger.info(f"✓ Corpus nettoyé sauvegardé dans: {output_corpus}")


if __name__ == "__main__":
    main()
    
    #lancer l'embeddings
    embeddings.main()