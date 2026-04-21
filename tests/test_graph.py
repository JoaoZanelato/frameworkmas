from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from state import AgentState, RouterOutput
from graphs.main_graph import route_to_specialist, build_graph


def _state(domain: str) -> AgentState:
    return {"question": "Teste", "domain": domain, "context": "", "generation": ""}


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
    mock_structured.invoke.return_value = RouterOutput(domain="AGRO")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="Carência de 12 meses para tratores.")

    graph = build_graph(llm=mock_llm)
    final = graph.invoke({
        "question": "Qual a carência do Pronaf Trator?",
        "domain": "",
        "context": "",
        "generation": "",
    })

    assert final["domain"] == "AGRO"
    assert "Normativa Agro" in final["context"]
    assert final["generation"] == "Carência de 12 meses para tratores."


def test_graph_routes_pf_question_end_to_end():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(domain="PF")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="Prazo de 72 meses para consignado.")

    graph = build_graph(llm=mock_llm)
    final = graph.invoke({
        "question": "Qual o prazo do consignado?",
        "domain": "",
        "context": "",
        "generation": "",
    })

    assert final["domain"] == "PF"
    assert "Normativa PF" in final["context"]
    assert final["generation"] == "Prazo de 72 meses para consignado."
