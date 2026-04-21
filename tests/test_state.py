import pytest
from pydantic import ValidationError
from state import AgentState, RouterOutput


def test_agent_state_has_all_required_keys():
    state: AgentState = {
        "question": "Qual a taxa do Pronaf?",
        "domain": "AGRO",
        "context": "Normativa Agro: carência de 12 meses.",
        "generation": "A carência é de 12 meses.",
    }
    assert state["question"] == "Qual a taxa do Pronaf?"
    assert state["domain"] == "AGRO"
    assert state["context"] == "Normativa Agro: carência de 12 meses."
    assert state["generation"] == "A carência é de 12 meses."


def test_router_output_validates_domain_field():
    output = RouterOutput(domain="AGRO")
    assert output.domain == "AGRO"


def test_router_output_raises_on_missing_domain():
    with pytest.raises(ValidationError):
        RouterOutput()
