import pytest
from pydantic import ValidationError
from state import AgentState, RouterOutput


def test_agent_state_has_all_required_keys():
    state: AgentState = {
        "question": "Qual a taxa do Pronaf?",
        "structured_query": "Qual a taxa de juros e prazo do Pronaf Mais Alimentos para financiamento de tratores?",
        "domain": "AGRO",
        "context": "# Pronaf — carência de 12 meses.",
        "generation": "A carência é de 12 meses.",
    }
    assert state["question"] == "Qual a taxa do Pronaf?"
    assert state["structured_query"].startswith("Qual a taxa")
    assert state["domain"] == "AGRO"
    assert state["generation"] == "A carência é de 12 meses."


def test_router_output_validates_domain_and_structured_query():
    output = RouterOutput(
        domain="AGRO",
        structured_query="Qual a taxa de juros do Pronaf Mais Alimentos para trator?",
    )
    assert output.domain == "AGRO"
    assert "Pronaf" in output.structured_query


def test_router_output_raises_on_missing_fields():
    with pytest.raises(ValidationError):
        RouterOutput()
