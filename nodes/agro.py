from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_AGRO_CONTEXT = (
    "Normativa Agro: A carência para financiamento de tratores via Pronaf Mais Alimentos "
    "é de 12 meses. A taxa de juros é de 5% a.a. e o prazo máximo de amortização é de 10 anos. "
    "Para custeio de safra, o prazo é de até 2 anos com taxa de 6% a.a."
)

_AGRO_SYSTEM_PROMPT = (
    "Você é um especialista em crédito rural da cooperativa. "
    "Use SOMENTE o contexto normativo abaixo para responder ao gerente. "
    "Seja objetivo e cite os valores exatos.\n\nContexto: {context}"
)


def make_agro_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """Fábrica do nó especialista em crédito rural (Agro)."""

    def agro_node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=_AGRO_SYSTEM_PROMPT.format(context=_AGRO_CONTEXT)),
            HumanMessage(content=state["question"]),
        ]
        response = llm.invoke(messages)
        return {
            "context": _AGRO_CONTEXT,
            "generation": response.content,
        }

    return agro_node
