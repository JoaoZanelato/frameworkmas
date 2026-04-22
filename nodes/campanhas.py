from pathlib import Path
from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_NORMATIVOS_DIR = Path(__file__).parent.parent / "normativos" / "campanhas"

_CAMPANHAS_SYSTEM_PROMPT = (
    "Você é um especialista em campanhas e produtos promocionais da cooperativa Sicredi. "
    "Use SOMENTE o contexto normativo abaixo para responder ao gerente. "
    "Seja objetivo, cite os valores exatos e referencie a campanha ou produto aplicável.\n\n"
    "Contexto:\n{context}"
)


def _load_context() -> str:
    """Carrega todos os .md do diretório campanhas em uma única string (uma vez no make_*_node)."""
    parts = []
    for md_file in sorted(_NORMATIVOS_DIR.glob("*.md")):
        parts.append(md_file.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)


def make_campanhas_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """Fábrica do nó especialista em campanhas e produtos promocionais. Contexto carregado uma vez no closure."""
    context = _load_context()

    def campanhas_node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=_CAMPANHAS_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=state["structured_query"]),
        ]
        response = llm.invoke(messages)
        return {
            "context": context,
            "generation": response.content,
        }

    return campanhas_node
