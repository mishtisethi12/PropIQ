"""
PropIQ - Semantic Document RAG Service with ChromaDB & MMR
Embeds real estate listing descriptions, tenancy lease agreements, and locality reports.
Provides metadata filtering by City and Locality, diversity-optimized retrieval via MMR,
and authoritative LLM answer generation with source document citations.
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    lambda_mult: float = 0.7,
    k: int = 4
) -> List[int]:
    """
    Computes Maximal Marginal Relevance (MMR) ranking.
    MMR = argmax_{d_i in R \\ S} [lambda * Sim(d_i, q) - (1 - lambda) * max_{d_j in S} Sim(d_i, d_j)]
    
    Args:
        query_embedding: Query vector shape (D,)
        doc_embeddings: Candidate document matrix shape (N, D)
        lambda_mult: Diversity tradeoff (1.0 = pure relevance, 0.0 = maximal diversity)
        k: Target number of diverse documents to select
    """
    if len(doc_embeddings) == 0:
        return []
    if len(doc_embeddings) <= k:
        return list(range(len(doc_embeddings)))

    # Normalize vectors for exact cosine similarity
    q_norm = np.linalg.norm(query_embedding)
    if q_norm > 0:
        q_vec = query_embedding / q_norm
    else:
        q_vec = query_embedding

    d_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    d_norms[d_norms == 0] = 1.0
    normed_docs = doc_embeddings / d_norms

    # Initial query similarity
    sim_to_query = np.dot(normed_docs, q_vec)

    # Greedily pick the highest similarity document first
    first_choice = int(np.argmax(sim_to_query))
    selected = [first_choice]
    unselected = [i for i in range(len(doc_embeddings)) if i != first_choice]

    while len(selected) < k and unselected:
        selected_matrix = normed_docs[selected]
        unselected_matrix = normed_docs[unselected]

        # Pairwise similarity with already chosen docs
        sim_to_selected = np.dot(unselected_matrix, selected_matrix.T)
        max_sim_to_selected = np.max(sim_to_selected, axis=1)

        unselected_query_sim = sim_to_query[unselected]
        mmr_scores = lambda_mult * unselected_query_sim - (1.0 - lambda_mult) * max_sim_to_selected

        best_unselected_idx = int(np.argmax(mmr_scores))
        chosen_doc_idx = unselected[best_unselected_idx]

        selected.append(chosen_doc_idx)
        unselected.pop(best_unselected_idx)

    return selected

class ChromaRAGService:
    def __init__(
        self,
        persist_dir: str = "data/warehouse/chroma_db",
        collection_name: str = "propiq_real_estate"
    ):
        self.persist_dir = os.path.join(BASE_DIR, persist_dir)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedder = None

    def _get_embedder(self):
        """Initializes all-MiniLM-L6-v2 embedding model using Chroma native ONNX engine"""
        if self._embedder is None:
            try:
                import chromadb.utils.embedding_functions as ef
                self._embedder = ef.ONNXMiniLM_L6_V2()
            except Exception as e:
                print(f"Warning loading ONNX embedding function: {e}")
                self._embedder = None
        return self._embedder

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates all-MiniLM-L6-v2 embeddings for a list of strings"""
        embedder = self._get_embedder()
        if embedder is not None:
            raw_embs = embedder(texts)
            return [list(map(float, vec)) for vec in raw_embs]
        else:
            # Deterministic dense vector fallback if ONNX engine cannot initialize
            dim = 384
            res = []
            for t in texts:
                np.random.seed(abs(hash(t)) % (2**31))
                vec = np.random.randn(dim)
                vec = vec / np.linalg.norm(vec)
                res.append(vec.tolist())
            return res

    def get_collection(self):
        """Lazily connects to ChromaDB collection"""
        if self._collection is None:
            import chromadb
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def ingest_knowledge_base(self, force_reindex: bool = False) -> Dict[str, int]:
        """
        Embeds property listings, lease agreements, and locality profiles into ChromaDB.
        """
        col = self.get_collection()
        existing_count = col.count()
        if existing_count > 0 and not force_reindex:
            return {"total_indexed": existing_count, "reindexed": False}

        if force_reindex and existing_count > 0:
            # Reset collection
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            col = self._collection

        documents = []
        metadatas = []
        ids = []

        # -------------------------------------------------------------
        # 1. Ingest Property Listings (Deduplicated canonical records)
        # -------------------------------------------------------------
        listings_file = os.path.join(BASE_DIR, "data/warehouse/fact_property_listings.parquet")
        loc_file = os.path.join(BASE_DIR, "data/warehouse/dim_location.parquet")
        type_file = os.path.join(BASE_DIR, "data/warehouse/dim_property_type.parquet")
        amen_file = os.path.join(BASE_DIR, "data/warehouse/dim_amenities.parquet")

        if os.path.exists(listings_file):
            df_fact = pd.read_parquet(listings_file)
            df_loc = pd.read_parquet(loc_file) if os.path.exists(loc_file) else None
            df_type = pd.read_parquet(type_file) if os.path.exists(type_file) else None
            df_amen = pd.read_parquet(amen_file) if os.path.exists(amen_file) else None

            # Merge dimensions cleanly
            if df_loc is not None:
                loc_subset = df_loc.drop(columns=["city"], errors="ignore")
                df_fact = df_fact.merge(loc_subset, on="location_key", how="left")
            if df_type is not None:
                df_fact = df_fact.merge(df_type, on="property_type_key", how="left")
            if df_amen is not None:
                df_fact = df_fact.merge(df_amen, on="amenity_key", how="left")

            # Sample representative listings across all cities and localities (up to 400 docs)
            sample_df = df_fact.groupby(["city", "locality"]).head(15).reset_index(drop=True)
            for idx, row in sample_df.iterrows():
                listing_id = str(row["listing_id"])
                city = str(row.get("city", "Unknown"))
                locality = str(row.get("locality", "Unknown"))
                bhk = int(row.get("bhk_count", 2))
                bhk_label = str(row.get("bhk_label", f"{bhk} BHK"))
                price = float(row.get("price_inr", 0))
                rent = float(row.get("rent_inr", 0))
                yield_pct = float(row.get("rental_yield_pct", 0))
                area = float(row.get("carpet_area_sqft", 0))
                furnishing = str(row.get("furnishing_status", "Semi-Furnished"))
                category = str(row.get("property_category", "Apartment"))
                
                amenities = []
                if row.get("has_parking", 0): amenities.append("Reserved Parking")
                if row.get("has_gym", 0): amenities.append("Fitness Gym")
                if row.get("has_swimming_pool", 0): amenities.append("Swimming Pool")
                if row.get("has_clubhouse", 0): amenities.append("Clubhouse")
                if row.get("has_security", 0): amenities.append("24/7 Security")
                if row.get("has_power_backup", 0): amenities.append("100% Power Backup")
                amen_str = ", ".join(amenities) if amenities else "Standard Infrastructure"

                doc_text = (
                    f"Verified Listing: {bhk_label} {category} in {locality}, {city}. "
                    f"Carpet Area: {area:,.0f} sq.ft. "
                    f"Configuration: {bhk} Bedrooms, {int(row.get('bathrooms', 2))} Bathrooms, {int(row.get('balconies', 1))} Balconies. "
                    f"Furnishing: {furnishing}. "
                    f"On-site Amenities: {amen_str}. "
                    f"Financials: Capital Price INR {price:,.0f} (INR {price/max(area,1):,.0f}/sq.ft), "
                    f"Expected Monthly Rent: INR {rent:,.0f}/month. "
                    f"Gross Rental Yield: {yield_pct:.2f}%. "
                    f"Transit Score: {row.get('transit_score', 'N/A')}/100, Safety Rating: {row.get('safety_score', 'N/A')}/100. "
                    f"Listing Portal: {row.get('source_portal', 'PropIQ Verified Matrix')}."
                )

                documents.append(doc_text)
                ids.append(f"prop_{listing_id}_{idx}")
                metadatas.append({
                    "doc_type": "listing_description",
                    "city": city,
                    "locality": locality,
                    "bhk_count": bhk,
                    "price_inr": price,
                    "rent_inr": rent,
                    "rental_yield_pct": yield_pct,
                    "title": f"{bhk_label} {category} in {locality}, {city}",
                    "source": f"Listing {listing_id}"
                })

        # -------------------------------------------------------------
        # 2. Ingest Tenancy Lease Agreements & Legal Clauses
        # -------------------------------------------------------------
        lease_files = [
            ("LEASE_AGREEMENT_STANDARD_NCR.md", "NCR", "All NCR"),
            ("LEASE_AGREEMENT_BANGALORE_MODEL.md", "Bangalore", "All Bangalore")
        ]
        for fname, default_city, default_loc in lease_files:
            fpath = os.path.join(BASE_DIR, "data/raw/scraped_text", fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Chunk by markdown sections / clauses
                sections = re.split(r"\n(?=## |\n### )", content)
                for s_idx, sec in enumerate(sections):
                    clean_sec = sec.strip()
                    if len(clean_sec) < 50:
                        continue
                    first_line = clean_sec.split("\n")[0].replace("#", "").strip()
                    doc_id = f"lease_{fname.replace('.md','')}_{s_idx}"

                    # If NCR, create metadata for Noida, Gurgaon, Delhi so city filters hit
                    target_cities = ["Noida", "Gurgaon", "Delhi", "Greater Noida"] if default_city == "NCR" else [default_city]
                    for c in target_cities:
                        documents.append(f"[{c} Tenancy Regulations - {first_line}]\n{clean_sec}")
                        ids.append(f"{doc_id}_{c}")
                        metadatas.append({
                            "doc_type": "lease_agreement",
                            "city": c,
                            "locality": default_loc,
                            "bhk_count": 0,
                            "price_inr": 0.0,
                            "rent_inr": 0.0,
                            "rental_yield_pct": 0.0,
                            "title": f"{fname.replace('.md','')} - {first_line}",
                            "source": fname
                        })

        # -------------------------------------------------------------
        # 3. Ingest Locality Profiles & Infrastructure Reports
        # -------------------------------------------------------------
        loc_reports = glob.glob(os.path.join(BASE_DIR, "data/raw/scraped_text/LOC_REPORT_*.md"))
        for lpath in loc_reports:
            fname = os.path.basename(lpath)
            # Format: LOC_REPORT_Noida_Sector_137.md
            parts = fname.replace("LOC_REPORT_", "").replace(".md", "").split("_", 1)
            city = parts[0] if len(parts) > 0 else "Unknown"
            locality = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

            with open(lpath, "r", encoding="utf-8") as lf:
                lcontent = lf.read()

            lsections = re.split(r"\n(?=## |\n### )", lcontent)
            for s_idx, sec in enumerate(lsections):
                clean_sec = sec.strip()
                if len(clean_sec) < 40:
                    continue
                header = clean_sec.split("\n")[0].replace("#", "").strip()

                documents.append(f"[{city} - {locality} Market Dossier - {header}]\n{clean_sec}")
                ids.append(f"loc_{fname.replace('.md','')}_{s_idx}")
                metadatas.append({
                    "doc_type": "locality_review",
                    "city": city,
                    "locality": locality,
                    "bhk_count": 0,
                    "price_inr": 0.0,
                    "rent_inr": 0.0,
                    "rental_yield_pct": 0.0,
                    "title": f"{city} {locality} Infrastructure: {header}",
                    "source": fname
                })

        # Batch embed and insert into ChromaDB
        print(f"Embedding {len(documents)} documents using all-MiniLM-L6-v2...")
        batch_size = 64
        for i in range(0, len(documents), batch_size):
            b_docs = documents[i:i+batch_size]
            b_ids = ids[i:i+batch_size]
            b_meta = metadatas[i:i+batch_size]
            b_embs = self.embed_texts(b_docs)

            col.add(
                documents=b_docs,
                embeddings=b_embs,
                metadatas=b_meta,
                ids=b_ids
            )

        print(f"[OK] Ingestion complete: {col.count()} documents indexed in ChromaDB.")
        return {"total_indexed": col.count(), "reindexed": True}

    def retrieve_with_mmr(
        self,
        query: str,
        city_filter: Optional[str] = None,
        locality_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        k: int = 4,
        lambda_mult: float = 0.7,
        candidate_pool: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Retrieves diverse, relevant documents from ChromaDB using Maximal Marginal Relevance (MMR).
        Applies metadata filters for city and locality.
        """
        col = self.get_collection()
        if col.count() == 0:
            self.ingest_knowledge_base()

        # Build ChromaDB metadata filter
        where_filters = []
        if city_filter and city_filter.strip() and city_filter.lower() not in ["all", "all cities"]:
            where_filters.append({"city": city_filter.strip()})
        if locality_filter and locality_filter.strip() and locality_filter.lower() not in ["all", "all localities"]:
            where_filters.append({"locality": locality_filter.strip()})
        if doc_type_filter and doc_type_filter.strip() and doc_type_filter.lower() not in ["all", "all types"]:
            where_filters.append({"doc_type": doc_type_filter.strip()})

        where = None
        if len(where_filters) == 1:
            where = where_filters[0]
        elif len(where_filters) > 1:
            where = {"$and": where_filters}

        # Embed query using all-MiniLM-L6-v2
        query_embs = self.embed_texts([query])
        q_vec = np.array(query_embs[0])

        # Fetch candidate pool from ChromaDB
        try:
            results = col.query(
                query_embeddings=query_embs,
                n_results=min(candidate_pool, max(col.count(), 1)),
                where=where,
                include=["documents", "metadatas", "embeddings", "distances"]
            )
        except Exception as query_err:
            print(f"Warning: Chroma where query failed ({query_err}), falling back without filter.")
            results = col.query(
                query_embeddings=query_embs,
                n_results=min(candidate_pool, max(col.count(), 1)),
                include=["documents", "metadatas", "embeddings", "distances"]
            )

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        raw_embs = results["embeddings"][0] if results["embeddings"] else []
        doc_ids = results["ids"][0] if results["ids"] else []

        if not docs:
            return []

        doc_matrix = np.array(raw_embs)

        # Apply MMR selection
        selected_indices = maximal_marginal_relevance(
            query_embedding=q_vec,
            doc_embeddings=doc_matrix,
            lambda_mult=lambda_mult,
            k=k
        )

        selected_records = []
        for rank, idx in enumerate(selected_indices, 1):
            d_norm = doc_matrix[idx] / (np.linalg.norm(doc_matrix[idx]) or 1.0)
            q_norm = q_vec / (np.linalg.norm(q_vec) or 1.0)
            cosine_sim = float(np.dot(d_norm, q_norm))

            selected_records.append({
                "id": doc_ids[idx],
                "rank": rank,
                "document": docs[idx],
                "metadata": metas[idx],
                "similarity_score": round(max(cosine_sim, 0.0), 4),
                "title": metas[idx].get("title", f"Document {doc_ids[idx]}"),
                "city": metas[idx].get("city", "N/A"),
                "locality": metas[idx].get("locality", "N/A"),
                "doc_type": metas[idx].get("doc_type", "general"),
                "source": metas[idx].get("source", "PropIQ Knowledge Base")
            })

        return selected_records

    def answer_query(
        self,
        query: str,
        city_filter: Optional[str] = None,
        locality_filter: Optional[str] = None,
        k: int = 4,
        lambda_mult: float = 0.7,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG:
        1. Retrieves diverse source documents via ChromaDB and MMR
        2. Generates comprehensive answer using Gemini LLM (or robust contextual synthesis)
        3. Returns generated text and source document payload
        """
        retrieved_docs = self.retrieve_with_mmr(
            query=query,
            city_filter=city_filter,
            locality_filter=locality_filter,
            k=k,
            lambda_mult=lambda_mult
        )

        llm_response = None
        key = api_key or os.environ.get("GEMINI_API_KEY")

        # Try Gemini API if key is available
        if key and retrieved_docs:
            try:
                import google.generativeai as genai
                genai.configure(api_key=key)
                model = genai.GenerativeModel("gemini-2.0-flash")

                context_blocks = []
                for doc in retrieved_docs:
                    m = doc["metadata"]
                    context_blocks.append(
                        f"--- SOURCE [{doc['rank']}]: {doc['title']} (Type: {doc['doc_type']}, City: {doc['city']}, Locality: {doc['locality']}) ---\n"
                        f"{doc['document']}\n"
                    )
                full_context = "\n".join(context_blocks)

                prompt = f"""
You are PropIQ GenAI Real Estate Research Assistant.
Answer the user's research question using ONLY the retrieved source documents below.
Synthesize listing details (prices, rents, yields, amenities) and lease terms (security deposit, lock-in, escalation).
Cite the source numbers [Source 1], [Source 2], etc. for each fact.
Format in clear Markdown with bullet points and bold highlights.
Do not use any emojis.

User Question: {query}

Retrieved Knowledge Base Documents:
{full_context}
"""
                resp = model.generate_content(prompt)
                llm_response = resp.text
            except Exception as e:
                print(f"Gemini API generation error: {e}. Falling back to offline synthesis.")

        # Offline contextual synthesis fallback
        if not llm_response:
            llm_response = self._synthesize_offline_answer(query, retrieved_docs)

        return {
            "query": query,
            "answer": llm_response,
            "sources": retrieved_docs,
            "city_filter": city_filter,
            "locality_filter": locality_filter,
            "mmr_lambda": lambda_mult
        }

    def _synthesize_offline_answer(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """High-fidelity contextual answer generator when LLM API key is not supplied"""
        if not docs:
            return (
                f"### Research Findings for '{query}'\n\n"
                "No matching documents found in the vector database with the current filter criteria. "
                "Try broadening your city or locality filters, or asking about standard NCR/Bangalore lease rules, "
                "rental yields, or specific localities like Sector 137, Sector 62, or Indiranagar."
            )

        lines = [f"### Research Analysis: '{query}'\n"]

        # Group by document type
        listings = [d for d in docs if d["doc_type"] == "listing_description"]
        leases = [d for d in docs if d["doc_type"] == "lease_agreement"]
        reviews = [d for d in docs if d["doc_type"] == "locality_review"]

        if listings:
            lines.append("#### Verified Property Opportunities & Pricing:\n")
            for l in listings:
                m = l["metadata"]
                price_fmt = f"INR {m.get('price_inr', 0):,.0f}" if m.get('price_inr') else "N/A"
                rent_fmt = f"INR {m.get('rent_inr', 0):,.0f}/mo" if m.get('rent_inr') else "N/A"
                yield_fmt = f"{m.get('rental_yield_pct', 0):.2f}%" if m.get('rental_yield_pct') else "N/A"
                lines.append(
                    f"- **{m.get('title', 'Property Listing')}** [{l['rank']}]: "
                    f"Acquisition Capital: **{price_fmt}** | Monthly Rent: **{rent_fmt}** | Gross Yield: **{yield_fmt}**."
                )
            lines.append("")

        if leases:
            lines.append("#### Statutory Tenancy Terms & Lease Regulations:\n")
            for ls in leases:
                m = ls["metadata"]
                snippet = ls["document"].replace("\n", " ")[:280]
                lines.append(f"- **{m.get('title', 'Tenancy Regulation')}** [{ls['rank']}]: {snippet}...")
            lines.append("")

        if reviews:
            lines.append("#### Locality Demographics & Infrastructure Insights:\n")
            for r in reviews:
                m = r["metadata"]
                snippet = r["document"].replace("\n", " ")[:280]
                lines.append(f"- **{m.get('title', 'Locality Profile')}** [{r['rank']}]: {snippet}...")
            lines.append("")

        lines.append("#### Citations & Retrieval Metadata:")
        for d in docs:
            lines.append(f"- `[{d['rank']}]` **{d['title']}** (Type: `{d['doc_type']}`, City: `{d['city']}`, Locality: `{d['locality']}`, MMR Score: `{d['similarity_score']}`)")

        return "\n".join(lines)
