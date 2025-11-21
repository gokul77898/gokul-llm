"""
Universal Text Extractor Module

Supports extraction from multiple file formats:
- PDF (with page tracking)
- TXT
- DOCX
- HTML
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class TextExtractor:
    """Universal text extractor supporting multiple file formats."""
    
    SUPPORTED_FORMATS = ['.pdf', '.txt', '.docx', '.html', '.htm']
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """
        Extract text from file (auto-detect format).
        
        Args:
            file_path: Path to file
            
        Returns:
            str: Extracted text
            
        Raises:
            ValueError: If file format not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}. Supported: {cls.SUPPORTED_FORMATS}")
        
        logger.info(f"Extracting text from {path.name} ({suffix})")
        
        if suffix == '.pdf':
            return cls._extract_pdf(file_path)
        elif suffix == '.txt':
            return cls._extract_txt(file_path)
        elif suffix == '.docx':
            return cls._extract_docx(file_path)
        elif suffix in ['.html', '.htm']:
            return cls._extract_html(file_path)
        else:
            raise ValueError(f"Handler not implemented for: {suffix}")
    
    @classmethod
    def extract_text_with_pages(cls, file_path: str) -> List[Tuple[int, str]]:
        """
        Extract text with page information (PDF only).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List[Tuple[int, str]]: List of (page_number, text) tuples
        """
        path = Path(file_path)
        
        if path.suffix.lower() != '.pdf':
            # For non-PDF files, return single page
            text = cls.extract_text(file_path)
            return [(1, text)]
        
        return cls._extract_pdf_pages(file_path)
    
    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        """Extract all text from PDF."""
        try:
            import pypdf
            
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text_parts = []
                
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"[Page {page_num}]\n{text}")
                
                return "\n\n".join(text_parts)
                
        except ImportError:
            logger.warning("pypdf not available, trying pdfplumber")
            return TextExtractor._extract_pdf_plumber(file_path)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    @staticmethod
    def _extract_pdf_plumber(file_path: str) -> str:
        """Extract PDF using pdfplumber (fallback)."""
        try:
            import pdfplumber
            
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        text_parts.append(f"[Page {page_num}]\n{text}")
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            raise ImportError("Neither pypdf nor pdfplumber available. Install with: pip install pypdf")
        except Exception as e:
            logger.error(f"PDF extraction with pdfplumber failed: {e}")
            raise
    
    @staticmethod
    def _extract_pdf_pages(file_path: str) -> List[Tuple[int, str]]:
        """Extract PDF with page numbers."""
        try:
            import pypdf
            
            pages = []
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        pages.append((page_num, text))
            
            logger.info(f"Extracted {len(pages)} pages from PDF")
            return pages
            
        except ImportError:
            logger.warning("pypdf not available, trying pdfplumber")
            return TextExtractor._extract_pdf_pages_plumber(file_path)
        except Exception as e:
            logger.error(f"PDF page extraction failed: {e}")
            raise
    
    @staticmethod
    def _extract_pdf_pages_plumber(file_path: str) -> List[Tuple[int, str]]:
        """Extract PDF pages using pdfplumber."""
        try:
            import pdfplumber
            
            pages = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append((page_num, text))
            
            return pages
            
        except ImportError:
            raise ImportError("Neither pypdf nor pdfplumber available")
        except Exception as e:
            logger.error(f"PDF page extraction with pdfplumber failed: {e}")
            raise
    
    @staticmethod
    def _extract_txt(file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            raise
    
    @staticmethod
    def _extract_docx(file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document
            
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(paragraphs)
            
        except ImportError:
            raise ImportError("python-docx not available. Install with: pip install python-docx")
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise
    
    @staticmethod
    def _extract_html(file_path: str) -> str:
        """Extract text from HTML file."""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                return text
                
        except ImportError:
            raise ImportError("beautifulsoup4 not available. Install with: pip install beautifulsoup4")
        except Exception as e:
            logger.error(f"HTML extraction failed: {e}")
            raise
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file format is supported."""
        suffix = Path(file_path).suffix.lower()
        return suffix in cls.SUPPORTED_FORMATS
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported file formats."""
        return cls.SUPPORTED_FORMATS.copy()


# Convenience function
def extract_text(file_path: str) -> str:
    """
    Extract text from file (convenience function).
    
    Args:
        file_path: Path to file
        
    Returns:
        str: Extracted text
    """
    return TextExtractor.extract_text(file_path)
