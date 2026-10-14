"""
PropIQ - Hybrid Retriever (SQL Analytics + Vector Documents)
Blends exact analytical calculations from the warehouse with unstructured narrative context.
"""

from typing import Dict, Any, List
import pandas as pd
from src.rag.sql_agent import SQLAnalyticsAgent
from src.rag.document_indexer import SemanticVectorIndex

class HybridRetriever:
    def __init__(self, duckdb_path: str, docs_dir: str):
        self.sql_agent = SQLAnalyticsAgent(duckdb_path)
        self.vector_index = SemanticVectorIndex()
        self.vector_index.build_index_from_directory(docs_dir)

    def retrieve(self, user_query: str) -> Dict[str, Any]:
        """
        Executes hybrid retrieval:
        1. Checks for SQL analytical intent
        2. Retrieves relevant document passages (locality guides, lease clauses)
        3. Returns combined context package
        """
        sql_query, intent = self.sql_agent.parse_intent_and_generate_sql(user_query)
        sql_df = None
        sql_data_dict = []

        if sql_query:
            try:
                sql_df = self.sql_agent.execute_sql(sql_query)
                sql_data_dict = sql_df.to_dict(orient="records")
            except Exception as e:
                print(f"Warning: SQL execution error: {e}")

        # Retrieve text chunks
        doc_chunks = self.vector_index.query_similar_chunks(user_query, top_k=4)

        return {
            "query": user_query,
            "intent": intent,
            "generated_sql": sql_query,
            "sql_results": sql_data_dict,
            "sql_dataframe": sql_df,
            "document_chunks": doc_chunks
        }
