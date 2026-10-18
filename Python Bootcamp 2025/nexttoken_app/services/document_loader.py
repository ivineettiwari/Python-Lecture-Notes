"""Extract plain text from uploaded files (no Streamlit dependency)."""
import hashlib
import io


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        import docx
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".doc"):
        raise ValueError("Legacy .doc is not supported. Save it as .docx and upload again.")
    raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")
