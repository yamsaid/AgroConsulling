import os
import re

# To avoid importing heavy/non-installed dependencies (gradio), we'll parse and
# extract only the `format_response_with_sources` function source from
# `frontend/app.py` and exec it into a local namespace.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_app_path = os.path.join(ROOT, "frontend", "app.py")
with open(frontend_app_path, "r", encoding="utf-8") as f:
    src = f.read()

# Extract the function using a regex that captures the def line and the body
m = re.search(
    r"def format_response_with_sources\(.*?:\n(\s+.+?)(?=\n\ndef|\nif __name__ == '__main__'|\Z)",
    src,
    flags=re.S,
)
if not m:
    raise RuntimeError(
        "Could not extract format_response_with_sources from frontend/app.py"
    )

func_src = "def format_response_with_sources(api_response: dict) -> str:\n" + m.group(1)

ns = {}
exec(func_src, ns)
format_response_with_sources = ns["format_response_with_sources"]

sample_api_response = {
    "success": True,
    "reponse": "Réponse de test pour vérifier l'affichage des sources.",
    "sources": [
        {
            "chunk_id": "2_pdf_chunk_6",
            "text": "Extrait de texte...",
            "source": "2_a68607be.pdf",
            "source_institution": "FAO",
            "organisme": "FAO",
            "chunk_index": 6,
            "total_chunks": 247,
            "source_url": "https://faolex.fao.org/docs/pdf/bkf198258.pdf",
            "pertinence": 0.7823172807693481,
        },
        {
            "chunk_id": "118_chunk_1",
            "text": "Autre extrait...",
            "source": "118_f157060e.pdf",
            "organisme": "MiLeCole",
            "source_url": "https://example.org/docs/118_f157060e.pdf",
            "pertinence": 0.7520576119422913,
        },
        {
            "chunk_id": "134_chunk_2",
            "text": "Extrait 3...",
            "source": "134_4a61f54f.pdf",
            "organisme": "FAO",
            "source_url": "https://example.org/docs/134_4a61f54f.pdf",
            "pertinence": 0.7470619678497314,
        },
    ],
}

if __name__ == "__main__":
    html = format_response_with_sources(sample_api_response)
    print("--- HTML Preview (console) ---")
    print(html)
    out_path = "test_sources_preview.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            '<!doctype html><html><head><meta charset="utf-8"><title>Preview</title></head><body>'
        )
        f.write(html)
        f.write("</body></html>")
    print(
        f"Wrote preview to {out_path}. Open it in a browser to see the rendered result."
    )
