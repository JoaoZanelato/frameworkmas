# Changelog

## [0.1.0] - 2026-04-21

### Added

#### Core Architecture
- `state.py` — `AgentState` TypedDict com os campos `question`, `domain`, `context` e `generation`; `RouterOutput` Pydantic para saída estruturada do roteador
- `graphs/main_graph.py` — `build_graph(llm)` monta e compila o `StateGraph` do LangGraph; `route_to_specialist` implementa a aresta condicional que direciona para o nó especialista correto com base no `domain`

#### Nodes
- `nodes/router.py` — `make_router_node(llm)`: classifica a intenção da pergunta em AGRO, PF ou PJ usando `with_structured_output` (zero ambiguidade no retorno do LLM)
- `nodes/agro.py` — `make_agro_node(llm)`: especialista em crédito rural com contexto RAG simulado (Pronaf, custeio agrícola)
- `nodes/pf.py` — `make_pf_node(llm)`: especialista em crédito pessoa física com contexto RAG simulado (consignado, veículos, cartão)
- `nodes/pj.py` — `make_pj_node(llm)`: especialista em crédito pessoa jurídica com contexto RAG simulado (capital de giro, recebíveis)

#### Infrastructure
- `main.py` — ponto de entrada CLI com saída formatada no terminal; executa o grafo com uma pergunta de teste sobre Pronaf
- `requirements.txt` — dependências do projeto: `langgraph`, `langchain`, `langchain-openai`, `langchain-anthropic`, `pydantic`, `python-dotenv`, `pytest`
- `pyproject.toml` — configuração do pytest com `pythonpath = ["."]`
- `.env.example` — template de variáveis de ambiente (suporte a OpenAI e Anthropic)

#### Tests
- `tests/test_state.py` — 3 testes unitários para `AgentState` e `RouterOutput`
- `tests/test_nodes.py` — 6 testes unitários para todos os nós (router + 3 especialistas) com mocks de LLM
- `tests/test_graph.py` — 7 testes para `route_to_specialist` (função pura) e integração end-to-end do grafo com mock de LLM

**Total: 16 testes, 0 falhas**
