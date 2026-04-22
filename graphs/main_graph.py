from typing import Literal

from langchain_core.language_models import BaseChatModel
import os

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from state import AgentState
from nodes.router import make_router_node
from nodes.agro import make_agro_node
from nodes.pf import make_pf_node
from nodes.pj import make_pj_node
from nodes.campanhas import make_campanhas_node


def route_to_specialist(
    state: AgentState,
) -> Literal["agro_node", "pf_node", "pj_node", "campanhas_node"]:
    """Aresta condicional: lê o domain no estado e devolve o nome do próximo nó."""
    mapping = {"AGRO": "agro_node", "PF": "pf_node", "PJ": "pj_node", "CAMPANHAS": "campanhas_node"}
    return mapping.get(state["domain"], "pj_node")


def build_graph(llm: BaseChatModel | None = None):
    """
    Monta e compila o grafo do sistema multiagente.
    Aceita um LLM externo para facilitar a injeção de mocks em testes.
    """
    if llm is None:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model="qwen2.5:7b", base_url=base_url, temperature=0.0)

    builder = StateGraph(AgentState)

    # ── Registrar nós ──────────────────────────────────────────────────────────
    builder.add_node("router_node", make_router_node(llm))
    builder.add_node("agro_node", make_agro_node(llm))
    builder.add_node("pf_node", make_pf_node(llm))
    builder.add_node("pj_node", make_pj_node(llm))
    builder.add_node("campanhas_node", make_campanhas_node(llm))

    # ── Definir fluxo (edges) ──────────────────────────────────────────────────
    builder.add_edge(START, "router_node")
    builder.add_conditional_edges("router_node", route_to_specialist)
    builder.add_edge("agro_node", END)
    builder.add_edge("pf_node", END)
    builder.add_edge("pj_node", END)
    builder.add_edge("campanhas_node", END)

    return builder.compile()
