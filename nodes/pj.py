from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_PJ_CONTEXT = (
    "Normativa PJ: Capital de giro para empresas associadas pode ser contratado "
    "com prazo de até 24 meses, garantia de recebíveis e taxa a partir de 1,8% a.m. "
    "Antecipação de recebíveis tem prazo máximo de 180 dias e taxa de 1,5% a.m."
)

_PJ_SYSTEM_PROMPT = (
    "Você é um especialista em crédito para pessoa jurídica da cooperativa. "
    "Use SOMENTE o contexto normativo abaixo para responder ao gerente. "
    "Seja objetivo e cite os valores exatos.\n\nContexto: {context}"
)


def make_pj_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """Fábrica do nó especialista em crédito empresarial (PJ)."""

    def pj_node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=_PJ_SYSTEM_PROMPT.format(context=_PJ_CONTEXT)),
            HumanMessage(content=state["question"]),
        ]
        response = llm.invoke(messages)
        return {
            "context": _PJ_CONTEXT,
            "generation": response.content,
        }

    return pj_node
