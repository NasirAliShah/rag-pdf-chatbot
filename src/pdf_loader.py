"""
PDF Loader Module
================
This module handles the first step of the RAG pipeline: PDF → Text

What it does:
- Reads PDF files and extracts all text content
- Preserves the structure and order of content from the PDF
- Returns raw text that will be processed further

Why it's needed:
- PDFs are binary files that need to be converted to readable text
- We use PyPDF library which is simple and reliable for text extraction
"""

from pypdf import PdfReader
from pathlib import Path
from typing import Optional


class PDFLoader:
    """Loads and extracts text from PDF files."""
    
    def __init__(self, file_path: str):
        """
        Initialize the PDF loader.
        
        Args:
            file_path: Path to the PDF file to load
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if self.file_path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF. Got: {self.file_path.suffix}")
    
    def load(self) -> str:
        """
        Extract all text from the PDF.
        
        Returns:
            Complete text content from all pages
        """
        pdf_reader = PdfReader(self.file_path)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
        
        return text
    
    def load_with_metadata(self) -> dict:
        """
        Extract text and metadata from the PDF.
        
        Returns:
            Dictionary containing:
            - 'text': Complete text content
            - 'num_pages': Total number of pages
            - 'filename': Name of the PDF file
        """
        pdf_reader = PdfReader(self.file_path)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
        
        return {
            'text': text,
            'num_pages': len(pdf_reader.pages),
            'filename': self.file_path.name
        }
