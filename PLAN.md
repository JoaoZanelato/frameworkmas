Atue como um Engenheiro de IA Sênior especialista em LangGraph e Python. Preciso criar o MVP (Minimum Viable Product) de um Sistema Multiagentes de Suporte para as agências da minha cooperativa de crédito.

**O Contexto do Negócio:**
O sistema recebe dúvidas dos gerentes das agências e usa um agente roteador para classificar a intenção e direcionar a pergunta para o agente especialista correto (Agro, PF ou PJ). O especialista então formula a resposta baseada em uma base de conhecimento.

**O Que Eu Preciso:**
Escreva o código Python completo e funcional utilizando o framework `langgraph` e `langchain`. O código deve ser didático, tipado e pronto para rodar no terminal.

**Estrutura da Arquitetura do LangGraph solicitada:**

1. **State (O Estado do Grafo):**
   Crie uma classe `TypedDict` chamada `AgentState` contendo:
   - `question` (str): A dúvida original do usuário.
   - `domain` (str): A classificação feita pelo roteador (AGRO, PF, PJ ou DESCONHECIDO).
   - `context` (str): O trecho do manual recuperado (simulação do RAG).
   - `generation` (str): A resposta final gerada pelo especialista.

2. **Node 1: Agente Roteador (`router_node`):**
   - Deve usar `with_structured_output` para obrigar o LLM a devolver um JSON simples com a chave `domain`.
   - Instrução: Leia a `question` e classifique estritamente entre "AGRO", "PF" ou "PJ".

3. **Nodes Especialistas (Simulação de RAG):**
   Crie três nós simples (`agro_node`, `pf_node`, `pj_node`). 
   - Para este MVP, não conecte a um banco vetorial. Simule o RAG inserindo uma string estática (hardcoded) com uma regra inventada para cada área. 
   - Exemplo no `agro_node`: context = "Normativa Agro: A carência para tratores Pronaf é de 12 meses."
   - Após definir o contexto, o nó deve chamar o LLM para gerar a resposta (`generation`) baseada nesse contexto e na pergunta.

4. **Arestas (Edges) e Fluxo:**
   - O ponto de entrada (`START`) vai para o `router_node`.
   - Crie uma função de roteamento condicional (`route_to_specialist`) que lê o `domain` no estado e direciona a aresta para o `agro_node`, `pf_node` ou `pj_node`.
   - Todos os nós especialistas terminam no `END`.

5. **LLM Base:**
   Utilize `ChatOpenAI` ou `ChatAnthropic` configurado de forma genérica para que eu apenas substitua a minha API Key no arquivo `.env`.

6. **Execução:**
   No final do script, inclua um bloco `if __name__ == "__main__":` instanciando o grafo e rodando uma pergunta de teste sobre financiamento de trator, para que eu veja o print do estado sendo atualizado no terminal.

Por favor, mantenha o código modular, com comentários claros explicando o fluxo da máquina de estado do LangGraph.