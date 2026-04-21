# MAS MVP LangGraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir o MVP funcional do Sistema Multiagentes de Suporte às agências da cooperativa, com um nó roteador classificador e três nós especialistas (Agro, PF, PJ) com RAG simulado, rodando via terminal.

**Architecture:** `AgentState` (TypedDict) transita entre nós via `StateGraph` do LangGraph. O `router_node` usa `with_structured_output` para classificar o domínio com zero ambiguidade; a aresta condicional `route_to_specialist` encaminha o estado ao nó especialista correto; cada especialista injeta um contexto estático (simulação de RAG) e chama o LLM para gerar a resposta final.

**Tech Stack:** Python 3.11+, LangGraph ≥ 0.2, LangChain ≥ 0.2, langchain-openai, pydantic v2, python-dotenv, pytest ≥ 8.

---

## File Structure

```
FrameworkMAS/
├── .env.example               # Template de variáveis de ambiente
├── requirements.txt           # Dependências Python
├── pyproject.toml             # Configuração do pytest (pythonpath)
├── state.py                   # AgentState TypedDict + RouterOutput Pydantic
├── nodes/
│   ├── __init__.py
│   ├── router.py              # make_router_node(llm) → router_node
│   ├── agro.py                # make_agro_node(llm) → agro_node
│   ├── pf.py                  # make_pf_node(llm) → pf_node
│   └── pj.py                  # make_pj_node(llm) → pj_node
├── graphs/
│   ├── __init__.py
│   └── main_graph.py          # build_graph(llm), route_to_specialist
├── main.py                    # Ponto de entrada __main__
└── tests/
    ├── test_state.py
    ├── test_nodes.py
    └── test_graph.py
```

---

## Task 1: Bootstrap do Projeto

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `nodes/__init__.py`
- Create: `graphs/__init__.py`

- [ ] **Step 1: Criar requirements.txt**

```text
langgraph>=0.2.0
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
langchain-core>=0.2.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Criar pyproject.toml**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 3: Criar .env.example**

```dotenv
# Escolha um provedor e preencha a chave correspondente
OPENAI_API_KEY=sk-proj-sua-chave-aqui
# ANTHROPIC_API_KEY=sk-ant-sua-chave-aqui
```

- [ ] **Step 4: Criar pastas e arquivos __init__.py vazios**

```bash
mkdir -p nodes graphs tests
touch nodes/__init__.py graphs/__init__.py
```

- [ ] **Step 5: Instalar dependências e copiar .env**

```bash
pip install -r requirements.txt
cp .env.example .env
# Edite .env e insira sua API key real antes de continuar
```

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt pyproject.toml .env.example nodes/__init__.py graphs/__init__.py
git commit -m "chore: bootstrap project structure"
```

---

## Task 2: AgentState e RouterOutput

**Files:**
- Create: `state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Escrever os testes que falharão**

```python
# tests/test_state.py
import pytest
from pydantic import ValidationError
from state import AgentState, RouterOutput


def test_agent_state_has_all_required_keys():
    state: AgentState = {
        "question": "Qual a taxa do Pronaf?",
        "domain": "AGRO",
        "context": "Normativa Agro: carência de 12 meses.",
        "generation": "A carência é de 12 meses.",
    }
    assert state["question"] == "Qual a taxa do Pronaf?"
    assert state["domain"] == "AGRO"
    assert state["context"] == "Normativa Agro: carência de 12 meses."
    assert state["generation"] == "A carência é de 12 meses."


def test_router_output_validates_domain_field():
    output = RouterOutput(domain="AGRO")
    assert output.domain == "AGRO"


def test_router_output_raises_on_missing_domain():
    with pytest.raises(ValidationError):
        RouterOutput()
```

- [ ] **Step 2: Rodar os testes para confirmar que falham**

```bash
pytest tests/test_state.py -v
```

Esperado: `FAILED` com `ModuleNotFoundError: No module named 'state'`

- [ ] **Step 3: Implementar state.py**

```python
# state.py
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
```

- [ ] **Step 4: Rodar os testes para confirmar que passam**

```bash
pytest tests/test_state.py -v
```

Esperado: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat: add AgentState TypedDict and RouterOutput Pydantic model"
```

---

## Task 3: Router Node

**Files:**
- Create: `nodes/router.py`
- Create: `tests/test_nodes.py` (parcial — apenas testes do router)

- [ ] **Step 1: Escrever os testes que falharão**

```python
# tests/test_nodes.py
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from state import AgentState, RouterOutput
from nodes.router import make_router_node
from nodes.agro import make_agro_node
from nodes.pf import make_pf_node
from nodes.pj import make_pj_node


def _base_state(question: str = "Teste", domain: str = "") -> AgentState:
    return {"question": question, "domain": domain, "context": "", "generation": ""}


# ── Router ──────────────────────────────────────────────────────────────────

def test_router_node_classifies_agro():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(domain="AGRO")
    mock_llm.with_structured_output.return_value = mock_structured

    node = make_router_node(mock_llm)
    result = node(_base_state("Qual a taxa do Pronaf Trator?"))

    assert result == {"domain": "AGRO"}
    mock_llm.with_structured_output.assert_called_once_with(RouterOutput)


def test_router_node_classifies_pf():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(domain="PF")
    mock_llm.with_structured_output.return_value = mock_structured

    node = make_router_node(mock_llm)
    result = node(_base_state("Quero financiar um carro"))

    assert result == {"domain": "PF"}


def test_router_node_classifies_pj():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(domain="PJ")
    mock_llm.with_structured_output.return_value = mock_structured

    node = make_router_node(mock_llm)
    result = node(_base_state("Capital de giro para empresa"))

    assert result == {"domain": "PJ"}
```

- [ ] **Step 2: Rodar os testes para confirmar que falham**

```bash
pytest tests/test_nodes.py::test_router_node_classifies_agro -v
```

Esperado: `FAILED` com `ModuleNotFoundError: No module named 'nodes.router'`

- [ ] **Step 3: Implementar nodes/router.py**

```python
# nodes/router.py
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
```

- [ ] **Step 4: Rodar os testes do router para confirmar que passam**

```bash
pytest tests/test_nodes.py::test_router_node_classifies_agro tests/test_nodes.py::test_router_node_classifies_pf tests/test_nodes.py::test_router_node_classifies_pj -v
```

Esperado: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add nodes/router.py tests/test_nodes.py
git commit -m "feat: add router_node with structured output classification"
```

---

## Task 4: Nós Especialistas (Agro, PF, PJ)

**Files:**
- Create: `nodes/agro.py`
- Create: `nodes/pf.py`
- Create: `nodes/pj.py`
- Modify: `tests/test_nodes.py` (adicionar testes dos especialistas — já incluídos no Step 1 abaixo)

- [ ] **Step 1: Adicionar testes dos especialistas ao tests/test_nodes.py**

Adicione ao final de `tests/test_nodes.py` (após os testes do router):

```python
# ── Especialistas ────────────────────────────────────────────────────────────

def test_agro_node_sets_context_and_generation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="A carência é de 12 meses.")

    node = make_agro_node(mock_llm)
    result = node(_base_state("Qual a carência do Pronaf Trator?", domain="AGRO"))

    assert "Normativa Agro" in result["context"]
    assert result["generation"] == "A carência é de 12 meses."


def test_pf_node_sets_context_and_generation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Prazo máximo de 72 meses.")

    node = make_pf_node(mock_llm)
    result = node(_base_state("Qual o prazo máximo do consignado?", domain="PF"))

    assert "Normativa PF" in result["context"]
    assert result["generation"] == "Prazo máximo de 72 meses."


def test_pj_node_sets_context_and_generation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Capital de giro em até 24 meses.")

    node = make_pj_node(mock_llm)
    result = node(_base_state("Como funciona capital de giro?", domain="PJ"))

    assert "Normativa PJ" in result["context"]
    assert result["generation"] == "Capital de giro em até 24 meses."
```

- [ ] **Step 2: Rodar os testes para confirmar que falham**

```bash
pytest tests/test_nodes.py::test_agro_node_sets_context_and_generation -v
```

Esperado: `FAILED` com `ModuleNotFoundError: No module named 'nodes.agro'`

- [ ] **Step 3: Implementar nodes/agro.py**

```python
# nodes/agro.py
from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_AGRO_CONTEXT = (
    "Normativa Agro: A carência para financiamento de tratores via Pronaf Mais Alimentos "
    "é de 12 meses. A taxa de juros é de 5% a.a. e o prazo máximo de amortização é de 10 anos. "
    "Para custeio de safra, o prazo é de até 2 anos com taxa de 6% a.a."
)

_AGRO_SYSTEM_PROMPT = (
    "Você é um especialista em crédito rural da cooperativa. "
    "Use SOMENTE o contexto normativo abaixo para responder ao gerente. "
    "Seja objetivo e cite os valores exatos.\n\nContexto: {context}"
)


def make_agro_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """Fábrica do nó especialista em crédito rural (Agro)."""

    def agro_node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=_AGRO_SYSTEM_PROMPT.format(context=_AGRO_CONTEXT)),
            HumanMessage(content=state["question"]),
        ]
        response = llm.invoke(messages)
        return {
            "context": _AGRO_CONTEXT,
            "generation": response.content,
        }

    return agro_node
```

- [ ] **Step 4: Implementar nodes/pf.py**

```python
# nodes/pf.py
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
```

- [ ] **Step 5: Implementar nodes/pj.py**

```python
# nodes/pj.py
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
```

- [ ] **Step 6: Rodar todos os testes de nós**

```bash
pytest tests/test_nodes.py -v
```

Esperado: `6 passed`

- [ ] **Step 7: Commit**

```bash
git add nodes/agro.py nodes/pf.py nodes/pj.py tests/test_nodes.py
git commit -m "feat: add specialist nodes (agro, pf, pj) with simulated RAG context"
```

---

## Task 5: Grafo Principal

**Files:**
- Create: `graphs/main_graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Escrever os testes que falharão**

```python
# tests/test_graph.py
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from state import AgentState, RouterOutput
from graphs.main_graph import route_to_specialist, build_graph


def _state(domain: str) -> AgentState:
    return {"question": "Teste", "domain": domain, "context": "", "generation": ""}


# ── route_to_specialist (função pura) ────────────────────────────────────────

def test_route_to_specialist_returns_agro_node():
    assert route_to_specialist(_state("AGRO")) == "agro_node"


def test_route_to_specialist_returns_pf_node():
    assert route_to_specialist(_state("PF")) == "pf_node"


def test_route_to_specialist_returns_pj_node():
    assert route_to_specialist(_state("PJ")) == "pj_node"


def test_route_to_specialist_unknown_domain_defaults_to_pj_node():
    assert route_to_specialist(_state("DESCONHECIDO")) == "pj_node"


# ── build_graph ───────────────────────────────────────────────────────────────

def test_build_graph_compiles_without_error():
    graph = build_graph(llm=MagicMock())
    assert graph is not None


def test_graph_routes_agro_question_end_to_end():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(domain="AGRO")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="Carência de 12 meses para tratores.")

    graph = build_graph(llm=mock_llm)
    final = graph.invoke({
        "question": "Qual a carência do Pronaf Trator?",
        "domain": "",
        "context": "",
        "generation": "",
    })

    assert final["domain"] == "AGRO"
    assert "Normativa Agro" in final["context"]
    assert final["generation"] == "Carência de 12 meses para tratores."


def test_graph_routes_pf_question_end_to_end():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(domain="PF")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="Prazo de 72 meses para consignado.")

    graph = build_graph(llm=mock_llm)
    final = graph.invoke({
        "question": "Qual o prazo do consignado?",
        "domain": "",
        "context": "",
        "generation": "",
    })

    assert final["domain"] == "PF"
    assert "Normativa PF" in final["context"]
    assert final["generation"] == "Prazo de 72 meses para consignado."
```

- [ ] **Step 2: Rodar os testes para confirmar que falham**

```bash
pytest tests/test_graph.py -v
```

Esperado: `FAILED` com `ModuleNotFoundError: No module named 'graphs.main_graph'`

- [ ] **Step 3: Implementar graphs/main_graph.py**

```python
# graphs/main_graph.py
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from state import AgentState
from nodes.router import make_router_node
from nodes.agro import make_agro_node
from nodes.pf import make_pf_node
from nodes.pj import make_pj_node


def route_to_specialist(
    state: AgentState,
) -> Literal["agro_node", "pf_node", "pj_node"]:
    """Aresta condicional: lê o domain no estado e devolve o nome do próximo nó."""
    mapping = {"AGRO": "agro_node", "PF": "pf_node", "PJ": "pj_node"}
    return mapping.get(state["domain"], "pj_node")


def build_graph(llm: BaseChatModel | None = None):
    """
    Monta e compila o grafo do sistema multiagente.
    Aceita um LLM externo (útil para injeção de mocks em testes).
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    builder = StateGraph(AgentState)

    # ── Registrar nós ──────────────────────────────────────────────────────────
    builder.add_node("router_node", make_router_node(llm))
    builder.add_node("agro_node", make_agro_node(llm))
    builder.add_node("pf_node", make_pf_node(llm))
    builder.add_node("pj_node", make_pj_node(llm))

    # ── Definir fluxo (edges) ──────────────────────────────────────────────────
    builder.add_edge(START, "router_node")
    builder.add_conditional_edges("router_node", route_to_specialist)
    builder.add_edge("agro_node", END)
    builder.add_edge("pf_node", END)
    builder.add_edge("pj_node", END)

    return builder.compile()
```

- [ ] **Step 4: Rodar todos os testes do grafo**

```bash
pytest tests/test_graph.py -v
```

Esperado: `8 passed`

- [ ] **Step 5: Rodar a suite completa**

```bash
pytest -v
```

Esperado: `17 passed` (3 state + 6 nodes + 8 graph)

- [ ] **Step 6: Commit**

```bash
git add graphs/main_graph.py tests/test_graph.py
git commit -m "feat: add main StateGraph with conditional routing to specialist nodes"
```

---

## Task 6: Ponto de Entrada (main.py)

**Files:**
- Create: `main.py`

> Nota: `main.py` não tem teste automatizado — é o ponto de integração final que requer uma API Key real. Valide executando manualmente.

- [ ] **Step 1: Implementar main.py**

```python
# main.py
"""
Ponto de entrada do Sistema Multiagente MAS.
Execução: python main.py
Requisito: arquivo .env com OPENAI_API_KEY preenchida.
"""
from dotenv import load_dotenv

load_dotenv()  # carrega .env ANTES de qualquer import que leia variáveis de ambiente

from graphs.main_graph import build_graph
from state import AgentState


def run(question: str) -> None:
    graph = build_graph()

    initial_state: AgentState = {
        "question": question,
        "domain": "",
        "context": "",
        "generation": "",
    }

    print("=" * 60)
    print(f"PERGUNTA: {question}")
    print("=" * 60)

    final_state: AgentState = graph.invoke(initial_state)

    print(f"DOMÍNIO CLASSIFICADO : {final_state['domain']}")
    print(f"CONTEXTO (RAG sim.)  : {final_state['context'][:80]}...")
    print("-" * 60)
    print(f"RESPOSTA FINAL:\n{final_state['generation']}")
    print("=" * 60)


if __name__ == "__main__":
    run("Qual é a carência para financiamento de trator pelo Pronaf Mais Alimentos?")
```

- [ ] **Step 2: Verificar que o arquivo .env tem a API Key real**

```bash
grep "OPENAI_API_KEY" .env
```

Esperado: linha com sua chave real (não o template `sk-proj-sua-chave-aqui`)

- [ ] **Step 3: Executar o sistema**

```bash
python main.py
```

Esperado (exemplo de saída):
```
============================================================
PERGUNTA: Qual é a carência para financiamento de trator pelo Pronaf Mais Alimentos?
============================================================
DOMÍNIO CLASSIFICADO : AGRO
CONTEXTO (RAG sim.)  : Normativa Agro: A carência para financiamento de tratores via Pro...
------------------------------------------------------------
RESPOSTA FINAL:
De acordo com as normativas da cooperativa, a carência para
financiamento de tratores via Pronaf Mais Alimentos é de 12 meses...
============================================================
```

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add main.py entry point with terminal output"
```

---

## Self-Review

### 1. Spec Coverage

| Requisito da spec | Task que implementa |
|---|---|
| `AgentState` TypedDict com `question`, `domain`, `context`, `generation` | Task 2 |
| `router_node` com `with_structured_output` retornando `domain` | Task 3 |
| `agro_node` com RAG simulado hardcoded + chamada LLM | Task 4 |
| `pf_node` com RAG simulado hardcoded + chamada LLM | Task 4 |
| `pj_node` com RAG simulado hardcoded + chamada LLM | Task 4 |
| Aresta condicional `route_to_specialist` | Task 5 |
| `START → router_node → Specialist → END` | Task 5 |
| `ChatOpenAI` com configuração via `.env` | Task 5 (build_graph) + Task 1 (.env.example) |
| `if __name__ == "__main__"` com pergunta de trator | Task 6 |

Nenhuma lacuna identificada.

### 2. Placeholder Scan

Nenhum placeholder encontrado. Todos os code blocks contêm código completo e executável.

### 3. Type Consistency

- `AgentState` definido em `state.py:Task2` — usado em todos os nós e no grafo com os mesmos nomes de campos: `question`, `domain`, `context`, `generation`.
- `RouterOutput` definido em `state.py:Task2` — usado em `nodes/router.py:Task3` e nos testes.
- Nomes de nós no grafo (`"router_node"`, `"agro_node"`, `"pf_node"`, `"pj_node"`) — consistentes entre `build_graph` (Task 5) e `route_to_specialist` (Task 5).
- `make_*_node(llm)` signature — consistente em todos os quatro nós (Tasks 3 e 4).
- `build_graph(llm=None)` — assinatura consistente entre implementação (Task 5) e testes (Task 5).
