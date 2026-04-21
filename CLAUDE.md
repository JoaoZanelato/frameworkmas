# Contexto do Projeto: Framework MAS (Multi-Agent System) para Atendimento 

## 1. Visão Geral e Objetivo
Este repositório contém a implementação *code-first* de um Sistema Multiagentes baseado em Grafos (LangGraph) para atuar como Tier 0 de suporte às agências da cooperativa.
O objetivo é reduzir o gargalo do backoffice e agilizar o atendimento na ponta, evitando que o gerente pause o relacionamento com o associado para tirar dúvidas de normativas.
Este projeto também servirá como Trabalho de Conclusão de Curso (TCC) em Análise e Desenvolvimento de Sistemas (previsão: 2027), exigindo alto rigor técnico, arquitetura modular e boas práticas de Engenharia de Software.

## 2. Restrições de Infraestrutura e Custos
- **Hospedagem:** O projeto rodará internamente no `devconsole` / servidores `cas`. Custos de infraestrutura são zero.
- **Custos Variáveis:** Apenas consumo de API (tokens). 
- **Privacidade (Fase 2):** A arquitetura deve ser agnóstica a modelos para permitir uma futura substituição de LLMs de mercado (OpenAI/Anthropic) por modelos *Open Source* rodando localmente, garantindo sigilo absoluto dos normativos.

## 3. Topologia do Grafo (O Conceito "Graphify")
Toda a lógica deve ser construída orientada a Grafos de Estado. 

### 3.1. Definição do Estado (`AgentState`)
O estado global que transita entre os nós deve conter estritamente:
- `question` (str): A pergunta original e bruta do gerente (imutável — preservada para auditoria).
- `structured_query` (str): Versão técnica e completa da pergunta, gerada pelo `router_node` via query rewriting.
- `domain` (str): O destino (AGRO, PF, PJ).
- `context` (str): O texto normativo recuperado do RAG (arquivos `.md`).
- `generation` (str): A resposta a ser devolvida à agência.

### 3.2. Nós (Nodes)
- `Router_Node`: **Dupla responsabilidade** — classifica o domínio E reformula a pergunta bruta em `structured_query` técnica e completa. Usa LLM com `with_structured_output`. **Não consulta o banco vetorial**. Esta etapa é obrigatória e garante que uma única chamada ao especialista seja suficiente para resposta precisa — evitando retries por input ambíguo.
- `Agro_Specialist_Node`: Recebe `structured_query` (não o input bruto), carrega contexto de `normativos/agro/*.md`, gera resposta técnica.
- `PF_Specialist_Node`: Idem para `normativos/pf/*.md` — crédito pessoal, veículos, cartões.
- `PJ_Specialist_Node`: Idem para `normativos/pj/*.md` — capital de giro, recebíveis.

### 3.2.1. Diretriz: Query Rewriting é obrigatório
O `router_node` sempre deve receber o input bruto e produzir uma `structured_query` completa antes de encaminhar. Perguntas malfeitas (`"e o trator?"`, `"carência?"`) devem ser expandidas para queries técnicas explícitas. Isso garante precisão na primeira chamada e elimina a necessidade de interação extra com o gerente.

### 3.2.2. Diretriz: RAG via Markdown (Fase 2)
Os nós especialistas carregam contexto normativo de arquivos `.md` organizados por domínio (`normativos/agro/`, `normativos/pf/`, `normativos/pj/`). O carregamento ocorre **uma vez** no `make_*_node(llm)` e é fechado no closure — zero I/O por requisição. Máximo 2–3 arquivos por pasta no MVP para performance aceitável com Ollama local.

### 3.2.3. Diretriz: Ambiente vs. Produção
- **Desenvolvimento/testes:** Ollama + `qwen2.5:7b` local (PC RTX 4060, `OLLAMA_BASE_URL` no `.env`).
- **Produção:** Provider externo de baixa latência (Groq, Claude Haiku ou similar) para TTFT aceitável em atendimento real. A troca é feita em uma linha — nenhum nó conhece o provider.

### 3.3. Arestas e Fluxo (Edges)
1. `START` -> `Router_Node`
2. `Router_Node` -> `Conditional_Edge` (Lê o campo `classified_domain`)
3. `Conditional_Edge` direciona para um dos `Specialist_Nodes`.
4. `Specialist_Nodes` -> `END`.

## 4. Stack Tecnológica e Padrões de Código
- **Linguagem:** Python 3.11+.
- **Orquestração:** LangGraph e LangChain (para abstração de chamadas de LLM).
- **Abordagem:** Estritamente *Code-First*. Nenhuma solução *drag-and-drop* (ex: Copilot Studio) deve ser utilizada para a lógica central.
- **Tipagem:** Uso obrigatório de *Type Hints* (Pydantic, TypedDict) em todas as funções.
- **APIs:** FastAPI para expor o motor de inferência.

## 5. Regras de Comportamento do Assistente (Claude)
Ao me auxiliar na escrita do código neste projeto, você deve seguir estas diretrizes:
1. **Mentalidade de Engenharia Sênior:** Foque na manutenibilidade, tratamento de erros e escalabilidade.
2. **Sem Alucinação de Bibliotecas:** Use apenas os métodos documentados mais recentes do `langgraph`.
3. **Modularidade:** Nunca me dê um arquivo monolítico de 500 linhas. Separe os grafos, os nós e as ferramentas de RAG em módulos (ex: `/nodes/router.py`, `/graphs/main_graph.py`).
4. **Resolução de Dores:** Lembre-se sempre que a prioridade é a precisão da resposta (para fortalecer a confiança da agência) e a eficiência do código.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
