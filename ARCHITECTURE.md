# Arquitetura do FrameworkMAS

Sistema Multiagente (MAS) baseado em grafos de estado para atendimento Tier 0 de suporte às agências de uma cooperativa de crédito. Implementado com LangGraph + Ollama como trabalho acadêmico (TCC — ADS, previsão 2027).

---

## 1. Problema e Objetivo

**Dor:** O gerente da agência precisa pausar o atendimento com o associado para ligar ao backoffice e tirar dúvidas sobre normativas de crédito (Pronaf, consignado, capital de giro etc.).

**Solução:** Um agente de IA local que classifica a pergunta e responde instantaneamente com base nos normativos oficiais, sem expor dados a serviços externos.

**Escopo do MVP:** Classificação de domínio + resposta com contexto simulado (RAG hardcoded). A Fase 2 substitui o contexto fixo por um banco vetorial real.

---

## 2. Topologia do Grafo

```
START
  │
  ▼
┌─────────────────┐
│   router_node   │  ← classifica a pergunta em AGRO | PF | PJ
└────────┬────────┘
         │  conditional_edge  (lê state["domain"])
         ├──── AGRO ──► agro_node ──► END
         ├──── PF   ──► pf_node   ──► END
         └──── PJ   ──► pj_node   ──► END
```

### Fluxo do Estado

```
AgentState
  question   →  [router_node]  →  domain
  domain     →  [conditional]  →  roteia para especialista
  question   →  [specialist]   →  context + generation
```

---

## 3. Modelo de Estado (`state.py`)

```python
class AgentState(TypedDict):
    question:   str  # pergunta original do gerente
    domain:     str  # AGRO | PF | PJ (preenchido pelo router)
    context:    str  # trecho normativo recuperado (RAG)
    generation: str  # resposta final ao gerente

class RouterOutput(BaseModel):
    domain: str      # schema Pydantic para saída estruturada
```

**Por que `TypedDict`?** LangGraph exige que o estado seja um dicionário tipado. `TypedDict` é o contrato formal entre os nós — cada nó recebe e devolve um subconjunto do estado.

**Por que `RouterOutput` Pydantic?** `with_structured_output()` precisa de um schema para forçar o LLM a retornar JSON válido. Pydantic valida em tempo de execução e lança `ValidationError` se o modelo alucinar um campo inexistente.

---

## 4. Componentes

### `nodes/router.py` — Classificador de Domínio

**Responsabilidade:** recebe a pergunta e devolve o domínio (`domain`).

**Por que `with_structured_output`?** Em vez de parsear texto livre, o LangChain instrui o modelo a retornar JSON com o schema de `RouterOutput`. Resultado determinístico, sem regex frágil.

```python
_ROUTER_SYSTEM_PROMPT = (
    "Você é um classificador de intenção de crédito cooperativo. "
    "Classifique estritamente em: AGRO (crédito rural, Pronaf, custeio agrícola), "
    "PF (crédito pessoal, veículos, cartões, consignado) ou "
    "PJ (capital de giro, recebíveis, empresas). "
    "Retorne apenas o campo 'domain' com o valor exato em maiúsculas."
)
```

**Padrão factory:** `make_router_node(llm)` retorna um closure que captura o LLM. Isso permite injetar um mock nos testes sem alterar o código de produção.

---

### `nodes/agro.py` | `nodes/pf.py` | `nodes/pj.py` — Especialistas

**Responsabilidade:** recebe a pergunta + contexto normativo, gera resposta técnica.

**Padrão RAG simulado (MVP):** O contexto normativo está hardcoded como string no próprio módulo. Na Fase 2, esse contexto virá de um banco vetorial (ChromaDB / FAISS) via similaridade semântica.

```
System prompt = "Use SOMENTE o contexto abaixo. Seja objetivo. Cite valores exatos."
               + contexto normativo injetado
Human message = pergunta do gerente
```

**Por que `llm.invoke()` simples (sem structured output)?** A resposta dos especialistas é texto livre — o gerente precisa de uma frase legível, não de JSON.

| Nó | Contexto normativo (MVP) |
|----|--------------------------|
| `agro_node` | Pronaf Mais Alimentos: carência 12m, juros 5% a.a., prazo 10 anos. Custeio: 2 anos, 6% a.a. |
| `pf_node` | Consignado: 72 meses, 2,5% a.m. Veículos: 60 meses, entrada 20%. Cartão: limite R$ 2.000. |
| `pj_node` | Capital de giro: 24 meses, 1,8% a.m. Antecipação recebíveis: 180 dias, 1,5% a.m. |

---

### `graphs/main_graph.py` — Montagem do Grafo

```python
def build_graph(llm: BaseChatModel | None = None):
    if llm is None:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model="qwen2.5:7b", base_url=base_url, temperature=0.0)
    ...
```

**Injeção de dependência:** `build_graph(llm=mock)` é o contrato dos testes. Sem isso, cada teste exigiria um servidor Ollama rodando.

**`route_to_specialist`:** função pura (sem LLM) que lê `state["domain"]` e devolve o nome do próximo nó. Domínio desconhecido cai em `pj_node` como default seguro.

---

## 5. Stack e Decisões de Infraestrutura

| Componente | Escolha | Motivo |
|-----------|---------|--------|
| Orquestração | LangGraph | Fluxo determinístico baseado em grafos de estado; ideal para MAS com roteamento condicional |
| LLM backend | Ollama + `qwen2.5:7b` | Zero custo, execução local, privacidade total dos normativos, sem dependência de API externa |
| LangChain | `langchain-ollama` | Abstração sobre o modelo; troca de provider sem alterar os nós |
| Tipagem | Pydantic v2 + TypedDict | Validação em runtime + contrato estático entre nós |
| Testes | pytest + MagicMock | 100% isolados, sem chamada real ao LLM |
| Python | 3.14 | Versão do ambiente; `langchain-google-genai` descartado por incompatibilidade de `protobuf` com Python 3.14 |

### Por que Gemini foi descartado

`langchain-google-genai` depende de `protobuf`, cuja extensão C (`google._upb._message`) usa metaclasses com `tp_new` customizado — não suportado no Python 3.14. O erro `TypeError: Metaclasses with custom tp_new are not supported` ocorre no import, antes de qualquer chamada à API.

### Configuração de servidor remoto

```
# .env
OLLAMA_BASE_URL=http://localhost:11434       # Ollama local
OLLAMA_BASE_URL=http://192.168.x.x:11434    # Ollama no PC com GPU (RTX 4060)
```

Para expor o Ollama na rede local (no PC com GPU):
```bash
sudo systemctl edit ollama
# adicionar:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

---

## 6. Testes

**Estratégia:** 100% mocks — nenhum teste faz chamada real ao LLM. Os testes validam a lógica do grafo e dos nós, não a qualidade das respostas do modelo.

```
tests/
  test_state.py   →  3 testes: AgentState, RouterOutput, ValidationError
  test_nodes.py   →  6 testes: router (3 domínios) + especialistas (3 nós)
  test_graph.py   →  7 testes: route_to_specialist (4) + build_graph + end-to-end (2)
```

**Padrão de mock:**
```python
mock_llm = MagicMock()
mock_structured = MagicMock()
mock_structured.invoke.return_value = RouterOutput(domain="AGRO")
mock_llm.with_structured_output.return_value = mock_structured
```

**Executar:**
```bash
pytest tests/ -v
# 16 passed in ~0.6s
```

---

## 7. Roadmap — Fase 2

| Item | Descrição |
|------|-----------|
| RAG real | Substituir contexto hardcoded por ChromaDB/FAISS com embeddings dos normativos |
| FastAPI | Endpoint `POST /query` que recebe `{ question, agencia_id }` e retorna `{ domain, generation }` |
| Deploy | Servidor `cas` interno da cooperativa com Ollama rodando `qwen2.5:7b` |
| Privacidade | Nenhum normativo trafega para fora da rede interna |
| Histórico | Persistência de conversas por agência para auditoria |

---

## 8. Estrutura de Arquivos

```
FrameworkMAS/
├── state.py                  # AgentState + RouterOutput
├── main.py                   # ponto de entrada (CLI)
├── graphs/
│   └── main_graph.py         # build_graph() + route_to_specialist()
├── nodes/
│   ├── router.py             # make_router_node()
│   ├── agro.py               # make_agro_node()
│   ├── pf.py                 # make_pf_node()
│   └── pj.py                 # make_pj_node()
├── tests/
│   ├── test_state.py
│   ├── test_nodes.py
│   └── test_graph.py
├── requirements.txt
├── pyproject.toml
└── .env                      # OLLAMA_BASE_URL
```

