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
