from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_PF_CONTEXT = (
    "Normativa PF: O prazo máximo para crédito pessoal consignado é de 72 meses, "
    "com taxa máxima de 2,5% a.m. Para financiamento de veículos, o prazo é de até 60 meses "
    "com entrada mínima de 20%. Cartão de crédito cooperativo tem limite inicial de R$ 2.000."
)

_PF_SYSTEM_PROMPT = (
    "Você é um especialista em crédito para pessoa física da cooperativa. "
    "Use SOMENTE o contexto normativo abaixo para responder ao gerente. "
    "Seja objetivo e cite os valores exatos.\n\nContexto: {context}"
)


def make_pf_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """Fábrica do nó especialista em crédito pessoal (PF)."""

    def pf_node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=_PF_SYSTEM_PROMPT.format(context=_PF_CONTEXT)),
            HumanMessage(content=state["question"]),
        ]
        response = llm.invoke(messages)
        return {
            "context": _PF_CONTEXT,
            "generation": response.content,
        }

    return pf_node
