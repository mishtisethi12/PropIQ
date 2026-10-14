"""
PropIQ - Semantic Document Indexer & Vector Retriever
Indexes unstructured listing notes, neighborhood profiles, and lease agreements.
Uses TF-IDF + Cosine Similarity Vector Space Model with zero external API dependency.
"""

import os
import glob
import re
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentChunk:
    def __init__(self, doc_id: str, chunk_id: str, title: str, text: str, doc_type: str):
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.title = title
        self.text = text
        self.doc_type = doc_type

class SemanticVectorIndex:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_df=0.85,
            min_df=1
        )
        self.chunks: List[DocumentChunk] = []
        self.tfidf_matrix = None
        self.is_indexed = False

    def chunk_document(self, text: str, title: str, doc_id: str, doc_type: str, chunk_size: int = 400) -> List[DocumentChunk]:
        """Splits document text into logical paragraphs or sentences"""
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        chunks = []
        for idx, p in enumerate(paragraphs):
            chunks.append(DocumentChunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{idx}",
                title=title,
                text=p,
                doc_type=doc_type
            ))
        return chunks

    def build_index_from_directory(self, docs_dir: str):
        """Builds index from markdown and text files in directory"""
        files = glob.glob(os.path.join(docs_dir, "*.*"))
        all_chunks = []

        for fpath in files:
            ext = os.path.splitext(fpath)[1].lower()
            if ext not in [".md", ".txt", ".json"]:
                continue
            fname = os.path.basename(fpath)
            doc_id = os.path.splitext(fname)[0]
            
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            doc_type = "lease_agreement" if "lease" in fname.lower() else "locality_profile"
            # Extract first line as title
            lines = content.split("\n")
            title = lines[0].replace("#", "").strip() if lines else doc_id

            chunks = self.chunk_document(content, title, doc_id, doc_type)
            all_chunks.extend(chunks)

        if not all_chunks:
            # Add fallback default document
            all_chunks.append(DocumentChunk(
                doc_id="DEFAULT_NCR",
                chunk_id="chunk_0",
                title="Delhi NCR Real Estate Overview",
                text="Noida and Gurgaon provide strong rental yields ranging between 3.5% and 5.2% for 2BHK units.",
                doc_type="locality_profile"
            ))

        self.chunks = all_chunks
        corpus_texts = [c.text for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.is_indexed = True

    def query_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Queries the vector index and returns top relevant chunks with similarity score"""
        if not self.is_indexed or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.02:  # Relevancy threshold
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "doc_type": chunk.doc_type,
                    "text": chunk.text,
                    "relevance_score": round(score, 4)
                })

        return results
