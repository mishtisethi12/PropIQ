import os
import pytest
from src.rag.assistant import PropIQAssistant
from src.rag.sql_agent import SQLAnalyticsAgent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUCKDB_PATH = os.path.join(BASE_DIR, "data/warehouse/propiq_warehouse.duckdb")
DOCS_DIR = os.path.join(BASE_DIR, "data/raw/scraped_text")

def test_sql_agent_intent_parsing():
    agent = SQLAnalyticsAgent(DUCKDB_PATH)
    sql, intent = agent.parse_intent_and_generate_sql("Which 2BHK areas in Noida have the best rental yield?")
    assert intent == "rental_yield_ranking"
    assert "fact_property_listings" in sql
    assert "Noida" in sql

def test_assistant_end_to_end_query():
    assistant = PropIQAssistant(DUCKDB_PATH, DOCS_DIR)
    res = assistant.answer_query("Which 2BHK areas in Noida have the best rental yield?")
    assert "answer" in res
    assert len(res["answer"]) > 50
    assert "Sector" in res["answer"]

def test_chroma_rag_mmr_retrieval():
    from src.rag.chroma_rag import ChromaRAGService, maximal_marginal_relevance
    import numpy as np

    # Test MMR algorithm function
    q = np.array([1.0, 0.0])
    docs = np.array([[0.9, 0.1], [0.89, 0.12], [0.1, 0.9]])
    mmr_idx = maximal_marginal_relevance(q, docs, lambda_mult=0.5, k=2)
    assert len(mmr_idx) == 2
    assert mmr_idx[0] == 0  # Most similar first

    # Test ChromaRAG service
    rag = ChromaRAGService()
    docs = rag.retrieve_with_mmr("lock in period deposit terms", city_filter="Noida", k=3)
    assert len(docs) <= 3
    assert len(docs) > 0
    assert "rank" in docs[0]
    assert "document" in docs[0]
    assert "metadata" in docs[0]

def test_chroma_rag_city_and_locality_filtering():
    from src.rag.chroma_rag import ChromaRAGService
    rag = ChromaRAGService()
    bangalore_docs = rag.retrieve_with_mmr("rent and lease deposit", city_filter="Bangalore", k=2)
    assert len(bangalore_docs) > 0
    for d in bangalore_docs:
        assert d["city"] == "Bangalore"

def test_chroma_rag_answer_and_sources_payload():
    from src.rag.chroma_rag import ChromaRAGService
    rag = ChromaRAGService()
    res = rag.answer_query("What are the lease rules in Noida?", city_filter="Noida", k=3)
    assert "answer" in res
    assert "sources" in res
    assert len(res["sources"]) > 0
    assert "title" in res["sources"][0]

