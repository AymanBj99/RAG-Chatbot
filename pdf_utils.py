import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file):
    """Extrait le texte d'un fichier PDF."""
    text = ""
    
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    
    return text.strip()
