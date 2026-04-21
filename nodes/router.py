from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState, RouterOutput

_ROUTER_SYSTEM_PROMPT = (
    "Você é um classificador de intenção de crédito cooperativo. "
    "Leia a pergunta do gerente e classifique estritamente em uma das categorias: "
    "AGRO (crédito rural, Pronaf, custeio agrícola), "
    "PF (crédito pessoal, veículos, cartões, consignado) ou "
    "PJ (capital de giro, recebíveis, empresas). "
    "Retorne apenas o campo 'domain' com o valor exato em maiúsculas."
)


def make_router_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """Fábrica que encapsula o LLM no closure do nó roteador."""

    def router_node(state: AgentState) -> dict:
        structured_llm = llm.with_structured_output(RouterOutput)
        messages = [
            SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]
        result: RouterOutput = structured_llm.invoke(messages)
        return {"domain": result.domain}

    return router_node
