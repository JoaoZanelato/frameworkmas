from typing import Any, Iterator, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from state import RouterOutput

_AGRO_KEYWORDS = {"pronaf", "trator", "agro", "rural", "safra", "custeio", "pecuário", "bovino", "soja", "milho", "carência", "finame rural", "seguro rural", "proagro", "agrícola", "agricultura", "fazenda", "lavoura"}
_PF_KEYWORDS = {"consignado", "veículo", "carro", "cartão", "pessoal", "pessoa física", "pf", "inss", "habitação", "imóvel", "previdência", "seguro de vida", "prestamista"}
_CAMPANHAS_KEYWORDS = {"campanha", "bem", "movimentação", "capitalização", "sicredi cap", "consórcio", "pontos", "sorteio", "prêmio", "lance", "carta de crédito"}


def _classify(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in _CAMPANHAS_KEYWORDS):
        return "CAMPANHAS"
    if any(k in lower for k in _AGRO_KEYWORDS):
        return "AGRO"
    if any(k in lower for k in _PF_KEYWORDS):
        return "PF"
    return "PJ"


def _make_structured_query(question: str, domain: str) -> str:
    templates = {
        "AGRO": f"[STUB-AGRO] Quais as condições (taxa, prazo, carência, garantia) para a operação de crédito rural referente a: {question}",
        "PF": f"[STUB-PF] Quais as condições (taxa, prazo, limite, requisitos) para a linha de crédito pessoal referente a: {question}",
        "PJ": f"[STUB-PJ] Quais as condições (taxa, prazo, garantia, limite) para a linha de crédito empresarial referente a: {question}",
        "CAMPANHAS": f"[STUB-CAMPANHAS] Quais as regras, condições e benefícios da campanha ou produto promocional referente a: {question}",
    }
    return templates[domain]


class _StructuredOutputWrapper:
    """Wrapper que simula with_structured_output para o RouterOutput."""

    def __init__(self, schema: type) -> None:
        self._schema = schema

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        question = next((m.content for m in messages if m.type == "human"), "")
        domain = _classify(question)
        return RouterOutput(
            domain=domain,
            structured_query=_make_structured_query(question, domain),
        )


class StubLLM(BaseChatModel):
    """LLM determinístico para testes sem conexão de rede."""

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages: List[BaseMessage], stop: Any = None, **kwargs: Any) -> ChatResult:
        query = next((m.content for m in messages if m.type == "human"), "pergunta desconhecida")
        preview = query[:60] + ("..." if len(query) > 60 else "")
        content = f"[STUB] Resposta gerada sem LLM para: {preview}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def with_structured_output(self, schema: type, **kwargs: Any) -> _StructuredOutputWrapper:  # type: ignore[override]
        return _StructuredOutputWrapper(schema)
