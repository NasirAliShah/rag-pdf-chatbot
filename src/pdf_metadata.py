"""
PDF Metadata Manager
====================
Tracks uploaded PDFs and their Pinecone namespaces.

Allows users to:
- See previously uploaded PDFs
- Load vectors from Pinecone without re-uploading
- Switch between different PDFs
- Delete PDFs from Pinecone
"""

import json
import os
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path


class PDFMetadata:
    """Stores metadata about uploaded PDFs."""
    
    def __init__(self, filename: str, namespace: str, num_chunks: int, upload_date: str):
        """
        Initialize PDF metadata.
        
        Args:
            filename: Name of PDF file
            namespace: Pinecone namespace where vectors are stored
            num_chunks: Number of chunks created
            upload_date: When PDF was uploaded
        """
        self.filename = filename
        self.namespace = namespace
        self.num_chunks = num_chunks
        self.upload_date = upload_date
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "namespace": self.namespace,
            "num_chunks": self.num_chunks,
            "upload_date": self.upload_date
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PDFMetadata":
        """Create from dictionary."""
        return cls(
            filename=data["filename"],
            namespace=data["namespace"],
            num_chunks=data["num_chunks"],
            upload_date=data["upload_date"]
        )


class PDFMetadataManager:
    """Manages PDF metadata storage and retrieval."""
    
    def __init__(self, metadata_file: str = ".pdf_metadata.json"):
        """
        Initialize metadata manager.
        
        Args:
            metadata_file: Path to JSON file storing metadata
        """
        self.metadata_file = metadata_file
        self.metadata: Dict[str, PDFMetadata] = {}
        self.load_metadata()
    
    def load_metadata(self) -> None:
        """Load metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    self.metadata = {
                        key: PDFMetadata.from_dict(value)
                        for key, value in data.items()
                    }
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def save_metadata(self) -> None:
        """Save metadata to file."""
        try:
            with open(self.metadata_file, "w") as f:
                data = {
                    key: value.to_dict()
                    for key, value in self.metadata.items()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def add_pdf(
        self,
        filename: str,
        namespace: str,
        num_chunks: int
    ) -> None:
        """
        Add PDF metadata.
        
        Args:
            filename: PDF filename
            namespace: Pinecone namespace
            num_chunks: Number of chunks
        """
        metadata = PDFMetadata(
            filename=filename,
            namespace=namespace,
            num_chunks=num_chunks,
            upload_date=datetime.now().isoformat()
        )
        
        # Use filename as key
        key = filename.replace(".pdf", "").replace(" ", "_")
        self.metadata[key] = metadata
        self.save_metadata()
    
    def get_pdf(self, key: str) -> Optional[PDFMetadata]:
        """Get PDF metadata by key."""
        return self.metadata.get(key)
    
    def list_pdfs(self) -> List[PDFMetadata]:
        """List all uploaded PDFs."""
        return list(self.metadata.values())
    
    def delete_pdf(self, key: str) -> bool:
        """Delete PDF metadata."""
        if key in self.metadata:
            del self.metadata[key]
            self.save_metadata()
            return True
        return False
    
    def get_namespace(self, filename: str) -> Optional[str]:
        """Get Pinecone namespace for PDF."""
        key = filename.replace(".pdf", "").replace(" ", "_")
        metadata = self.get_pdf(key)
        return metadata.namespace if metadata else None
    
    def get_summary(self) -> dict:
        """Get summary of all PDFs."""
        return {
            "total_pdfs": len(self.metadata),
            "pdfs": [
                {
                    "filename": m.filename,
                    "chunks": m.num_chunks,
                    "uploaded": m.upload_date
                }
                for m in self.metadata.values()
            ]
        }
