from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from state import AgentState, RouterOutput
from nodes.router import make_router_node
from nodes.agro import make_agro_node
from nodes.pf import make_pf_node
from nodes.pj import make_pj_node


def _base_state(question: str = "Teste", domain: str = "") -> AgentState:
    return {"question": question, "domain": domain, "context": "", "generation": ""}


# ── Router ───────────────────────────────────────────────────────────────────

def test_router_node_classifies_agro():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(domain="AGRO")
    mock_llm.with_structured_output.return_value = mock_structured

    node = make_router_node(mock_llm)
    result = node(_base_state("Qual a taxa do Pronaf Trator?"))

    assert result == {"domain": "AGRO"}
    mock_llm.with_structured_output.assert_called_once_with(RouterOutput)


def test_router_node_classifies_pf():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(domain="PF")
    mock_llm.with_structured_output.return_value = mock_structured

    node = make_router_node(mock_llm)
    result = node(_base_state("Quero financiar um carro"))

    assert result == {"domain": "PF"}


def test_router_node_classifies_pj():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(domain="PJ")
    mock_llm.with_structured_output.return_value = mock_structured

    node = make_router_node(mock_llm)
    result = node(_base_state("Capital de giro para empresa"))

    assert result == {"domain": "PJ"}


# ── Especialistas ────────────────────────────────────────────────────────────

def test_agro_node_sets_context_and_generation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="A carência é de 12 meses.")

    node = make_agro_node(mock_llm)
    result = node(_base_state("Qual a carência do Pronaf Trator?", domain="AGRO"))

    assert "Normativa Agro" in result["context"]
    assert result["generation"] == "A carência é de 12 meses."


def test_pf_node_sets_context_and_generation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Prazo máximo de 72 meses.")

    node = make_pf_node(mock_llm)
    result = node(_base_state("Qual o prazo máximo do consignado?", domain="PF"))

    assert "Normativa PF" in result["context"]
    assert result["generation"] == "Prazo máximo de 72 meses."


def test_pj_node_sets_context_and_generation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Capital de giro em até 24 meses.")

    node = make_pj_node(mock_llm)
    result = node(_base_state("Como funciona capital de giro?", domain="PJ"))

    assert "Normativa PJ" in result["context"]
    assert result["generation"] == "Capital de giro em até 24 meses."
