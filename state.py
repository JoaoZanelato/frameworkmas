from typing import Literal, TypedDict
from pydantic import BaseModel


class AgentState(TypedDict):
    """Estado global que transita entre todos os nós do grafo."""
    question: str           # Dúvida original do gerente (imutável — preservada para auditoria)
    structured_query: str   # Query técnica reescrita pelo router_node via query rewriting
    domain: str             # Classificação: AGRO | PF | PJ | CAMPANHAS
    context: str            # Texto normativo recuperado dos arquivos .md (RAG)
    generation: str         # Resposta final gerada pelo nó especialista


class RouterOutput(BaseModel):
    """Schema obrigatório para saída estruturada do nó roteador."""
    domain: Literal["AGRO", "PF", "PJ", "CAMPANHAS"]
    structured_query: str
