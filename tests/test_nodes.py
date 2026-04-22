from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from state import AgentState, RouterOutput
from nodes.router import make_router_node
from nodes.agro import make_agro_node
from nodes.pf import make_pf_node
from nodes.pj import make_pj_node


def _base_state(question: str = "Teste", structured_query: str = "Query técnica de teste.", domain: str = "") -> AgentState:
    return {"question": question, "structured_query": structured_query, "domain": domain, "context": "", "generation": ""}


# ── Router ───────────────────────────────────────────────────────────────────

def test_router_node_classifies_agro():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(
        domain="AGRO",
        structured_query="Qual a carência e taxa do Pronaf Mais Alimentos para financiamento de tratores?",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    node = make_router_node(mock_llm)
    result = node(_base_state("Qual a taxa do Pronaf Trator?"))

    assert result["domain"] == "AGRO"
    assert "structured_query" in result
    mock_llm.with_structured_output.assert_called_once_with(RouterOutput)


def test_router_node_classifies_pf():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(
        domain="PF",
        structured_query="Qual o prazo máximo e taxa para financiamento de veículo leve (carro) para pessoa física?",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    node = make_router_node(mock_llm)
    result = node(_base_state("Quero financiar um carro"))

    assert result["domain"] == "PF"
    assert "structured_query" in result


def test_router_node_classifies_pj():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(
        domain="PJ",
        structured_query="Quais as condições (prazo, taxa, garantia) para contratação de capital de giro para pessoa jurídica associada?",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    node = make_router_node(mock_llm)
    result = node(_base_state("Capital de giro para empresa"))

    assert result["domain"] == "PJ"
    assert "structured_query" in result


def test_router_node_returns_both_domain_and_structured_query():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    sq = "Qual a taxa de juros do Pronaf Custeio para safra de soja?"
    mock_structured.invoke.return_value = RouterOutput(domain="AGRO", structured_query=sq)
    mock_llm.with_structured_output.return_value = mock_structured

    node = make_router_node(mock_llm)
    result = node(_base_state("taxa custeio soja"))

    assert result == {"domain": "AGRO", "structured_query": sq}


# ── Especialistas ────────────────────────────────────────────────────────────

def test_agro_node_sets_context_and_generation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="A carência é de 12 meses.")

    node = make_agro_node(mock_llm)
    result = node(_base_state(
        question="Qual a carência do Pronaf Trator?",
        structured_query="Qual a carência para financiamento de tratores via Pronaf Mais Alimentos?",
        domain="AGRO",
    ))

    assert "Pronaf" in result["context"]
    assert result["generation"] == "A carência é de 12 meses."


def test_pf_node_sets_context_and_generation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Prazo máximo de 72 meses.")

    node = make_pf_node(mock_llm)
    result = node(_base_state(
        question="Qual o prazo máximo do consignado?",
        structured_query="Qual o prazo máximo e taxa para crédito consignado com desconto em folha para pessoa física?",
        domain="PF",
    ))

    assert "consignado" in result["context"].lower() or "crédito" in result["context"].lower()
    assert result["generation"] == "Prazo máximo de 72 meses."


def test_pj_node_sets_context_and_generation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Capital de giro em até 24 meses.")

    node = make_pj_node(mock_llm)
    result = node(_base_state(
        question="Como funciona capital de giro?",
        structured_query="Quais as condições de prazo, taxa e garantia para capital de giro para pessoa jurídica?",
        domain="PJ",
    ))

    assert "capital de giro" in result["context"].lower() or "recebíveis" in result["context"].lower()
    assert result["generation"] == "Capital de giro em até 24 meses."


def test_specialist_node_uses_structured_query_not_raw_question():
    """O nó especialista deve passar structured_query ao LLM, não question."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Resposta técnica.")

    node = make_agro_node(mock_llm)
    state = _base_state(
        question="e o trator?",
        structured_query="Qual a carência e taxa para financiamento de tratores via Pronaf Mais Alimentos?",
        domain="AGRO",
    )
    node(state)

    call_args = mock_llm.invoke.call_args[0][0]
    human_message_content = call_args[1].content
    assert human_message_content == state["structured_query"]
    assert human_message_content != state["question"]
