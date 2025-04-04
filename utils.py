from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    """Extrait le texte d'un fichier PDF"""
    reader = PdfReader(file_path)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text
