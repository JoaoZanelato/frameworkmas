from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from state import AgentState, RouterOutput
from graphs.main_graph import route_to_specialist, build_graph


def _state(domain: str) -> AgentState:
    return {
        "question": "Teste",
        "structured_query": "Query técnica de teste.",
        "domain": domain,
        "context": "",
        "generation": "",
    }


def _initial_state(question: str) -> dict:
    return {"question": question, "structured_query": "", "domain": "", "context": "", "generation": ""}


# ── route_to_specialist (função pura — sem LLM, sem mock) ────────────────────

def test_route_to_specialist_returns_agro_node():
    assert route_to_specialist(_state("AGRO")) == "agro_node"


def test_route_to_specialist_returns_pf_node():
    assert route_to_specialist(_state("PF")) == "pf_node"


def test_route_to_specialist_returns_pj_node():
    assert route_to_specialist(_state("PJ")) == "pj_node"


def test_route_to_specialist_unknown_domain_defaults_to_pj_node():
    assert route_to_specialist(_state("DESCONHECIDO")) == "pj_node"


# ── build_graph ───────────────────────────────────────────────────────────────

def test_build_graph_compiles_without_error():
    graph = build_graph(llm=MagicMock())
    assert graph is not None


def test_graph_routes_agro_question_end_to_end():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    sq = "Qual a carência para financiamento de tratores via Pronaf Mais Alimentos?"
    mock_structured.invoke.return_value = RouterOutput(domain="AGRO", structured_query=sq)
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="Carência de 12 meses para tratores.")

    graph = build_graph(llm=mock_llm)
    final = graph.invoke(_initial_state("Qual a carência do Pronaf Trator?"))

    assert final["domain"] == "AGRO"
    assert final["structured_query"] == sq
    assert "Pronaf" in final["context"]
    assert final["generation"] == "Carência de 12 meses para tratores."


def test_graph_routes_pf_question_end_to_end():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    sq = "Qual o prazo máximo e taxa para crédito consignado com desconto em folha para pessoa física?"
    mock_structured.invoke.return_value = RouterOutput(domain="PF", structured_query=sq)
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="Prazo de 72 meses para consignado.")

    graph = build_graph(llm=mock_llm)
    final = graph.invoke(_initial_state("Qual o prazo do consignado?"))

    assert final["domain"] == "PF"
    assert final["structured_query"] == sq
    assert "consignado" in final["context"].lower() or "crédito" in final["context"].lower()
    assert final["generation"] == "Prazo de 72 meses para consignado."


def test_graph_routes_pj_question_end_to_end():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    sq = "Quais as condições de prazo, taxa e garantia para capital de giro para pessoa jurídica associada?"
    mock_structured.invoke.return_value = RouterOutput(domain="PJ", structured_query=sq)
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="Capital de giro em até 24 meses.")

    graph = build_graph(llm=mock_llm)
    final = graph.invoke(_initial_state("Como funciona capital de giro para empresa?"))

    assert final["domain"] == "PJ"
    assert final["structured_query"] == sq
    assert "capital de giro" in final["context"].lower()
    assert final["generation"] == "Capital de giro em até 24 meses."
