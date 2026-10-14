"""
PropIQ - Scraper & Document Ingestion Engine
Ingests unstructured real estate text, landlord notes, HTML files, and lease contracts.
"""

import os
import glob
from typing import List, Dict, Any
from bs4 import BeautifulSoup

class ScraperIngestionEngine:
    def __init__(self, scraped_dir: str):
        self.scraped_dir = scraped_dir

    def extract_text_from_html(self, html_content: str) -> str:
        """Strips HTML tags and extracts clean textual content"""
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n").strip()

    def load_scraped_documents(self) -> List[Dict[str, Any]]:
        """Scans scraped text directory and parses all listing/lease markdown and HTML docs"""
        documents = []
        if not os.path.exists(self.scraped_dir):
            return documents

        files = glob.glob(os.path.join(self.scraped_dir, "*.*"))
        for filepath in files:
            ext = os.path.splitext(filepath)[1].lower()
            filename = os.path.basename(filepath)
            
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if ext in [".html", ".htm"]:
                clean_text = self.extract_text_from_html(content)
            else:
                clean_text = content.strip()

            doc_type = "lease_agreement" if "lease" in filename.lower() else "locality_profile"

            documents.append({
                "doc_id": filename.replace(ext, ""),
                "file_path": filepath,
                "doc_type": doc_type,
                "text_content": clean_text,
                "character_count": len(clean_text)
            })

        return documents
