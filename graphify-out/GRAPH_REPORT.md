# Graph Report - /home/joaozanelato/Documentos/FrameworkMAS  (2026-04-21)

## Corpus Check
- 12 files · ~6,585 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 50 nodes · 81 edges · 9 communities detected
- Extraction: 59% EXTRACTED · 41% INFERRED · 0% AMBIGUOUS · INFERRED: 33 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]

## God Nodes (most connected - your core abstractions)
1. `RouterOutput` - 11 edges
2. `AgentState` - 10 edges
3. `build_graph()` - 10 edges
4. `_base_state()` - 7 edges
5. `make_router_node()` - 6 edges
6. `route_to_specialist()` - 6 edges
7. `_state()` - 5 edges
8. `make_agro_node()` - 4 edges
9. `make_pf_node()` - 4 edges
10. `make_pj_node()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `Ponto de entrada do Sistema Multiagente MAS. Execução: python main.py Requisito:` --uses--> `AgentState`  [INFERRED]
  /home/joaozanelato/Documentos/FrameworkMAS/main.py → /home/joaozanelato/Documentos/FrameworkMAS/state.py
- `Fábrica do nó especialista em crédito rural (Agro).` --uses--> `AgentState`  [INFERRED]
  /home/joaozanelato/Documentos/FrameworkMAS/nodes/agro.py → /home/joaozanelato/Documentos/FrameworkMAS/state.py
- `Fábrica do nó especialista em crédito empresarial (PJ).` --uses--> `AgentState`  [INFERRED]
  /home/joaozanelato/Documentos/FrameworkMAS/nodes/pj.py → /home/joaozanelato/Documentos/FrameworkMAS/state.py
- `Monta e compila o grafo do sistema multiagente.     Aceita um LLM externo para f` --uses--> `AgentState`  [INFERRED]
  /home/joaozanelato/Documentos/FrameworkMAS/graphs/main_graph.py → /home/joaozanelato/Documentos/FrameworkMAS/state.py
- `Fábrica que encapsula o LLM no closure do nó roteador.` --uses--> `AgentState`  [INFERRED]
  /home/joaozanelato/Documentos/FrameworkMAS/nodes/router.py → /home/joaozanelato/Documentos/FrameworkMAS/state.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.33
Nodes (9): make_router_node(), Fábrica que encapsula o LLM no closure do nó roteador., _base_state(), test_agro_node_sets_context_and_generation(), test_pf_node_sets_context_and_generation(), test_pj_node_sets_context_and_generation(), test_router_node_classifies_agro(), test_router_node_classifies_pf() (+1 more)

### Community 1 - "Community 1"
Cohesion: 0.22
Nodes (7): build_graph(), Monta e compila o grafo do sistema multiagente.     Aceita um LLM externo para f, Ponto de entrada do Sistema Multiagente MAS. Execução: python main.py Requisito:, run(), test_build_graph_compiles_without_error(), test_graph_routes_agro_question_end_to_end(), test_graph_routes_pf_question_end_to_end()

### Community 2 - "Community 2"
Cohesion: 0.25
Nodes (6): Aresta condicional: lê o domain no estado e devolve o nome do próximo nó., make_pf_node(), Fábrica do nó especialista em crédito pessoal (PF)., AgentState, Estado global que transita entre todos os nós do grafo., TypedDict

### Community 3 - "Community 3"
Cohesion: 0.33
Nodes (5): BaseModel, Schema obrigatório para saída estruturada do nó roteador., RouterOutput, test_router_output_raises_on_missing_domain(), test_router_output_validates_domain_field()

### Community 4 - "Community 4"
Cohesion: 0.62
Nodes (6): route_to_specialist(), _state(), test_route_to_specialist_returns_agro_node(), test_route_to_specialist_returns_pf_node(), test_route_to_specialist_returns_pj_node(), test_route_to_specialist_unknown_domain_defaults_to_pj_node()

### Community 5 - "Community 5"
Cohesion: 0.67
Nodes (2): make_pj_node(), Fábrica do nó especialista em crédito empresarial (PJ).

### Community 6 - "Community 6"
Cohesion: 0.67
Nodes (2): make_agro_node(), Fábrica do nó especialista em crédito rural (Agro).

### Community 7 - "Community 7"
Cohesion: 1.0
Nodes (0): 

### Community 8 - "Community 8"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **2 isolated node(s):** `Estado global que transita entre todos os nós do grafo.`, `Schema obrigatório para saída estruturada do nó roteador.`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 7`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 8`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_graph()` connect `Community 1` to `Community 0`, `Community 2`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.320) - this node is a cross-community bridge._
- **Why does `RouterOutput` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`?**
  _High betweenness centrality (0.298) - this node is a cross-community bridge._
- **Why does `AgentState` connect `Community 2` to `Community 0`, `Community 1`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.223) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `RouterOutput` (e.g. with `Fábrica que encapsula o LLM no closure do nó roteador.` and `test_router_output_validates_domain_field()`) actually correct?**
  _`RouterOutput` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `AgentState` (e.g. with `Ponto de entrada do Sistema Multiagente MAS. Execução: python main.py Requisito:` and `Fábrica que encapsula o LLM no closure do nó roteador.`) actually correct?**
  _`AgentState` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `build_graph()` (e.g. with `run()` and `make_router_node()`) actually correct?**
  _`build_graph()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `make_router_node()` (e.g. with `build_graph()` and `test_router_node_classifies_agro()`) actually correct?**
  _`make_router_node()` has 4 INFERRED edges - model-reasoned connections that need verification._