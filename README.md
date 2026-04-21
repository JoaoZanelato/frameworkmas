# FrameworkMAS

Sistema Multiagentes de Suporte para agências de cooperativa de crédito, construído com LangGraph.

## Arquitetura

```
Pergunta do Gerente
       │
       ▼
  router_node  ──── with_structured_output ────► classifica domain (AGRO | PF | PJ)
       │
       ▼ (aresta condicional)
 ┌─────┴──────┬──────────────┐
 ▼            ▼              ▼
agro_node   pf_node       pj_node
(RAG Agro)  (RAG PF)     (RAG PJ)
 └─────┬──────┴──────────────┘
       │
       ▼
  Resposta Final
```

## Estrutura

```
FrameworkMAS/
├── state.py              # AgentState TypedDict + RouterOutput Pydantic
├── nodes/
│   ├── router.py         # Classificador de domínio
│   ├── agro.py           # Especialista crédito rural
│   ├── pf.py             # Especialista crédito pessoal
│   └── pj.py             # Especialista crédito empresarial
├── graphs/
│   └── main_graph.py     # StateGraph (build_graph + route_to_specialist)
└── main.py               # Entrada CLI
```

## Instalação

```bash
pip install -r requirements.txt
cp .env.example .env
# Edite .env e insira sua OPENAI_API_KEY
```

## Uso

```bash
python main.py
```

## Testes

```bash
pytest -v   # 16 testes, sem chamadas de API
```
