"""
Ponto de entrada do Sistema Multiagente MAS.

Uso:
  python main.py "Qual a carência do Pronaf Trator?"          # Ollama remoto
  python main.py --stub "Qual a carência do Pronaf Trator?"   # sem LLM, sem rede
"""
import argparse
from dotenv import load_dotenv

load_dotenv()

from graphs.main_graph import build_graph
from state import AgentState


def run(question: str, use_stub: bool = False) -> None:
    if use_stub:
        from stubs.stub_llm import StubLLM
        llm = StubLLM()
        graph = build_graph(llm=llm)
    else:
        graph = build_graph()

    initial_state: AgentState = {
        "question": question,
        "structured_query": "",
        "domain": "",
        "context": "",
        "generation": "",
    }

    print("=" * 60)
    print(f"PERGUNTA          : {question}")
    print("=" * 60)

    final_state: AgentState = graph.invoke(initial_state)

    print(f"DOMÍNIO           : {final_state['domain']}")
    print(f"\nSTRUCTURED QUERY  :\n{final_state['structured_query']}")
    print(f"\nCONTEXTO (RAG)    :\n{final_state['context']}")
    print("-" * 60)
    print(f"RESPOSTA FINAL:\n{final_state['generation']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FrameworkMAS — Sistema Multiagente de Crédito")
    parser.add_argument("question", nargs="?", default="Qual é a carência para financiamento de trator pelo Pronaf Mais Alimentos?")
    parser.add_argument("--stub", action="store_true", help="Usar StubLLM (sem Ollama, sem rede)")
    args = parser.parse_args()

    run(args.question, use_stub=args.stub)
