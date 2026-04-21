"""
Ponto de entrada do Sistema Multiagente MAS.
Execução: python main.py
Requisito: variável OLLAMA_BASE_URL no .env (padrão: http://localhost:11434).
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
