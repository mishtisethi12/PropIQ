"""
PropIQ - GenAI Real Estate Assistant
Executes conversational question-answering over warehouse facts, locality reviews, and lease contracts.
Works out-of-the-box with built-in contextual synthesis and seamlessly connects with Gemini LLM.
"""

import os
from typing import Dict, Any, Optional
from src.rag.hybrid_retriever import HybridRetriever

class PropIQAssistant:
    def __init__(self, duckdb_path: str, docs_dir: str, api_key: Optional[str] = None):
        self.retriever = HybridRetriever(duckdb_path, docs_dir)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def _generate_rule_based_synthesis(self, retrieval_pkg: Dict[str, Any]) -> str:
        """
        High-fidelity offline synthesis engine:
        Combines SQL table outputs, locality reports, and lease contract rules.
        """
        query = retrieval_pkg["query"]
        sql_results = retrieval_pkg["sql_results"]
        doc_chunks = retrieval_pkg["document_chunks"]
        intent = retrieval_pkg["intent"]

        lines = []

        if intent == "rental_yield_ranking" and sql_results:
            top_rec = sql_results[0]
            lines.append(f"### Rental Yield Analysis: '{query}'\n")
            lines.append(f"Based on real estate data warehouse analytics, **{top_rec['locality']}** in **{top_rec['city']}** delivers the highest average rental yield at **{top_rec['avg_rental_yield_pct']}%** for {top_rec['bhk_label']} units.\n")
            
            lines.append("#### Top Performing Localities Ranking:\n")
            lines.append("| Rank | Locality | City | Configuration | Avg Price (INR) | Avg Monthly Rent (INR) | Rental Yield | Transit Score |")
            lines.append("|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|")
            for idx, r in enumerate(sql_results[:5], 1):
                price_fmt = f"INR {r['avg_price']:,.0f}"
                rent_fmt = f"INR {r['avg_rent']:,.0f}"
                lines.append(f"| {idx} | **{r['locality']}** | {r['city']} | {r['bhk_label']} | {price_fmt} | {rent_fmt} | **{r['avg_rental_yield_pct']}%** | {r.get('transit_score', 'N/A')}/100 |")

            lines.append("\n#### Investment Metrics & Insights:")
            lines.append(f"- **Top Opportunity**: {top_rec['locality']} offers the optimal balance of acquisition cost (INR {top_rec['avg_price']:,.0f}) and monthly rental cash flow (INR {top_rec['avg_rent']:,.0f}/month).")
            lines.append(f"- **Gross Rent Multiplier (GRM)**: Approximately {round(top_rec['avg_price'] / max(top_rec['avg_rent'] * 12, 1), 1)} years to recover asset cost via gross rent.")

        elif intent == "price_ranking" and sql_results:
            cheapest = sql_results[0]
            lines.append(f"### Capital Value Ranking: '{query}'\n")
            lines.append(f"The most competitive entry pricing is in **{cheapest['locality']}**, {cheapest['city']} with an average listing price of **INR {cheapest['avg_price']:,.0f}**.\n")
            lines.append("| Rank | Locality | City | Avg Price (INR) | Price/sq.ft | Rental Yield |")
            lines.append("|:---:|:---|:---|:---:|:---:|:---:|")
            for idx, r in enumerate(sql_results[:5], 1):
                lines.append(f"| {idx} | **{r['locality']}** | {r['city']} | INR {r['avg_price']:,.0f} | INR {r['avg_price_sqft']:,.0f} | {r['avg_rental_yield_pct']}% |")

        # Document Context
        if doc_chunks:
            lines.append("\n#### Locality Profile & Statutory Lease Regulations:")
            for chunk in doc_chunks[:2]:
                title = chunk["title"]
                text_snippet = chunk["text"]
                lines.append(f"**Reference Document [{title}]**:")
                lines.append(f"> {text_snippet}\n")

        if not lines:
            lines.append(f"### Market Intelligence Query\n")
            lines.append(f"Evaluated data warehouse records and document index for: '{query}'.\n")
            if doc_chunks:
                for chunk in doc_chunks:
                    lines.append(f"**{chunk['title']}**\n{chunk['text']}\n")
            else:
                lines.append("No direct records found matching this exact filter criteria. Specify target metro areas (e.g., Noida, Gurgaon, Delhi, Bangalore) or configuration (e.g., 2 BHK).")

        return "\n".join(lines)

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        """Processes user question using Hybrid RAG (SQL + Documents + LLM)"""
        retrieval_pkg = self.retriever.retrieve(user_query)

        # Try Gemini API if key is available
        llm_answer = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")

                # Build prompt
                sql_summary = str(retrieval_pkg["sql_results"]) if retrieval_pkg["sql_results"] else "No SQL query needed"
                doc_texts = "\n---\n".join([f"[{c['title']}]: {c['text']}" for c in retrieval_pkg["document_chunks"]])

                prompt = f"""
You are PropIQ, an elite Real Estate Data Engineering and Investment Copilot.
Answer the user's question clearly, professionally, and authoritatively using the retrieved facts below.

User Question: {user_query}

Verified Star Schema Warehouse Data (Exact metrics):
{sql_summary}

Retrieved Knowledge Base Documents (Lease agreements & Locality reviews):
{doc_texts}

Instructions:
1. When discussing rental yields, cite the exact percentages from the warehouse data.
2. If lease questions are asked, cite relevant clauses (lock-in, security deposit, escalation).
3. Format output cleanly in Markdown with tables and bullet points.
"""
                response = model.generate_content(prompt)
                llm_answer = response.text
            except Exception as e:
                print(f"Gemini API call skipped or encountered error: {e}. Falling back to contextual synthesis.")

        if not llm_answer:
            llm_answer = self._generate_rule_based_synthesis(retrieval_pkg)

        return {
            "answer": llm_answer,
            "sql_query": retrieval_pkg["generated_sql"],
            "sql_results": retrieval_pkg["sql_results"],
            "citations": [c["title"] for c in retrieval_pkg["document_chunks"]],
            "intent": retrieval_pkg["intent"]
        }
