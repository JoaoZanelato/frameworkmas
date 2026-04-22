from pathlib import Path
from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_NORMATIVOS_DIR = Path(__file__).parent.parent / "normativos" / "agro"

_AGRO_SYSTEM_PROMPT = (
    "Você é um especialista em crédito rural da cooperativa. "
    "Use SOMENTE o contexto normativo abaixo para responder ao gerente. "
    "Seja objetivo, cite os valores exatos e referencie o normativo aplicável.\n\n"
    "Contexto:\n{context}"
)


def _load_context() -> str:
    """Carrega todos os .md do diretório agro em uma única string (uma vez no make_*_node)."""
    parts = []
    for md_file in sorted(_NORMATIVOS_DIR.glob("*.md")):
        parts.append(md_file.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)


def make_agro_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """Fábrica do nó especialista em crédito rural (Agro). Contexto carregado uma vez no closure."""
    context = _load_context()

    def agro_node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=_AGRO_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=state["structured_query"]),
        ]
        response = llm.invoke(messages)
        return {
            "context": context,
            "generation": response.content,
        }

    return agro_node
