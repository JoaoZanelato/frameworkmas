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
- `user_query` (str): A pergunta original (ex: "Qual a taxa do Pronaf?").
- `metadata` (dict): Informações da requisição (agência, perfil).
- `classified_domain` (str): O destino (AGRO, PF, PJ, OFF_TOPIC).
- `retrieved_context` (str): O texto extraído do RAG.
- `final_answer` (str): A resposta a ser devolvida à agência.

### 3.2. Nós (Nodes)
- `Router_Node`: Otimizador e classificador. Usa LLM rápido. Retorna JSON estruturado. **Não consulta o banco vetorial**.
- `Agro_Specialist_Node`: Recebe a *query*, faz RAG nas normativas rurais, gera resposta técnica.
- `PF_Specialist_Node`: Especialista em crédito pessoal, veículos, cartões.
- `PJ_Specialist_Node`: Especialista em capital de giro, recebíveis.

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