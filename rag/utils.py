import fitz  # PyMuPDF
import uuid
import time
from typing import List, Tuple

def timer():
    """Simple decorator for measuring function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return result, round(end - start, 4)
        return wrapper
    return decorator

def extract_text_by_page(file_path: str) -> List[Tuple[int, str]]:
    """Extracts text from each page of a PDF."""
    doc = fitz.open(file_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            pages.append((page_num + 1, text))
    return pages

def generate_uuid() -> str:
    return str(uuid.uuid4())
