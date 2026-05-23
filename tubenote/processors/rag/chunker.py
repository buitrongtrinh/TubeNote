"""Text splitter — port nguyên từ rag.ipynb."""
from __future__ import annotations


def chunk_text(full_text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """Split → cleanup → Document objects."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_text(full_text)
    chunks = [c.lstrip(". ").strip() for c in chunks if c.strip()]
    return [Document(page_content=c) for c in chunks]
