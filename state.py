from typing import TypedDict
from pydantic import BaseModel


class AgentState(TypedDict):
    """Estado global que transita entre todos os nós do grafo."""
    question: str    # Dúvida original do gerente da agência
    domain: str      # Classificação: AGRO | PF | PJ
    context: str     # Trecho do normativo recuperado (RAG simulado)
    generation: str  # Resposta final gerada pelo especialista


class RouterOutput(BaseModel):
    """Schema obrigatório para saída estruturada do nó roteador."""
    domain: str
