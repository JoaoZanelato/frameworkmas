from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState, RouterOutput

_ROUTER_SYSTEM_PROMPT = (
    "Você é um classificador e reescritor de perguntas de crédito cooperativo. "
    "Receba a pergunta bruta do gerente e execute DUAS tarefas obrigatórias:\n\n"
    "1. CLASSIFICAR em exatamente um domínio:\n"
    "   - AGRO: crédito rural, Pronaf, custeio agrícola/pecuário, máquinas rurais.\n"
    "   - PF: crédito pessoal, consignado, veículos, cartão de crédito, financiamento pessoal.\n"
    "   - PJ: capital de giro, recebíveis, duplicatas, crédito empresarial, CNPJ.\n\n"
    "2. REESCREVER a pergunta em uma structured_query técnica e completa. "
    "Expanda abreviações, resolva ambiguidades e inclua todos os parâmetros relevantes "
    "que o especialista precisará para responder com precisão na primeira tentativa. "
    "Perguntas curtas como 'e o trator?' ou 'carência?' devem virar queries explícitas "
    "com produto, finalidade e parâmetro solicitado.\n\n"
    "Retorne os campos 'domain' e 'structured_query'."
)


def make_router_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """Fábrica do nó roteador: classifica domínio e reescreve a query."""

    def router_node(state: AgentState) -> dict:
        structured_llm = llm.with_structured_output(RouterOutput)
        messages = [
            SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]
        result: RouterOutput = structured_llm.invoke(messages)
        return {
            "domain": result.domain,
            "structured_query": result.structured_query,
        }

    return router_node
