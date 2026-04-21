# Design Spec: RAG via Markdown + Query Rewriting no Router

**Data:** 2026-04-21  
**Status:** Aprovado  
**Fase:** 2 — RAG real (substitui contexto hardcoded)

---

## 1. Problema

### 1.1 Contexto hardcoded (Fase 1)
Os nós especialistas (`agro_node`, `pf_node`, `pj_node`) carregam normativas como strings literais no código Python. Qualquer atualização de normativa exige alteração de código e redeploy.

### 1.2 Query malfeita (gap de qualidade)
O gerente de agência frequentemente digita perguntas ambíguas ou incompletas (`"e o trator?"`, `"carência?"`, `"quanto fica?"`). O nó especialista recebia esse input bruto e produzia respostas imprecisas ou incompletas, forçando o gerente a reformular — gerando chamadas extras ao LLM.

---

## 2. Solução

Duas mudanças ortogonais que se complementam:

1. **RAG via Markdown:** substituir strings hardcoded por arquivos `.md` carregados do sistema de arquivos.
2. **Query Rewriting no Router:** o `router_node` passa a ter dupla responsabilidade — classificar o domínio **e** reformular a pergunta bruta em query técnica e completa antes de encaminhar ao especialista.

---

## 3. Mudanças no Estado (`state.py`)

```python
class AgentState(TypedDict):
    question:         str  # input bruto do gerente (imutável — preservado para auditoria)
    structured_query: str  # query reformulada pelo router (nova)
    domain:           str  # AGRO | PF | PJ
    context:          str  # contexto MD concatenado pelo nó especialista
    generation:       str  # resposta final ao gerente

class RouterOutput(BaseModel):
    domain:           str  # classificação do domínio
    structured_query: str  # reformulação técnica da pergunta (nova)
```

**Por que preservar `question`?** O input bruto é mantido para fins de auditoria e rastreabilidade. O especialista usa `structured_query`; `question` fica disponível para logging e histórico futuro.

---

## 4. Router Node — Dupla Responsabilidade

### Diretriz de projeto (obrigatória)
> O `router_node` deve sempre receber o input bruto do gerente, classificar o domínio e reformular a pergunta em query técnica e completa antes de encaminhar ao nó especialista. Isso garante que **uma única chamada ao especialista seja suficiente** para gerar resposta precisa, evitando retries por input ambíguo.

### Novo system prompt

```
Você é classificador e reformulador de perguntas de crédito cooperativo.

Dado o input do gerente:
1. Classifique o domínio estritamente em: AGRO (crédito rural, Pronaf, custeio
   agrícola), PF (crédito pessoal, veículos, cartões, consignado) ou PJ (capital
   de giro, recebíveis, empresas).
2. Reescreva a pergunta de forma técnica, completa e sem ambiguidade, como se
   fosse feita por um analista de crédito. Expanda abreviações, resolva pronomes
   ambíguos e inclua o produto de crédito explicitamente.

Retorne JSON com os campos 'domain' e 'structured_query'.
```

### Exemplos de transformação

| Input bruto (gerente) | `structured_query` gerada pelo router |
|-----------------------|---------------------------------------|
| "e o trator?" | "Qual é a carência, taxa de juros e prazo máximo para financiamento de tratores via Pronaf Mais Alimentos?" |
| "carência?" | "Qual é o período de carência para o produto de crédito rural discutido anteriormente?" |
| "quanto fica o consignado?" | "Qual é a taxa de juros mensal e o prazo máximo para crédito consignado para pessoa física?" |

---

## 5. RAG Loader (`rag/loader.py`)

```python
def load_md_context(folder: str) -> str:
    """Lê todos os .md de uma pasta e retorna string concatenada."""
```

- Chamado **uma vez** dentro de `make_*_node(llm)`, no momento de construção do grafo.
- O contexto resultante fica em memória no closure — zero I/O por requisição.
- Arquivos são ordenados alfabeticamente para resultado determinístico.
- Se a pasta estiver vazia ou não existir, lança `FileNotFoundError` com mensagem clara.

---

## 6. Specialist Nodes — Mudança de Input

Os nós especialistas passam a usar `state["structured_query"]` no lugar de `state["question"]`:

```python
# Antes (Fase 1)
HumanMessage(content=state["question"])

# Depois (Fase 2)
HumanMessage(content=state["structured_query"])
```

O contexto normativo vem do `load_md_context` fechado no closure, substituindo `_AGRO_CONTEXT` hardcoded.

---

## 7. Estrutura de Arquivos Normativos

```
normativos/
├── agro/
│   ├── pronaf.md          # Pronaf Mais Alimentos — carência, juros, prazos
│   └── custeio_safra.md   # Custeio de safra — prazos e taxas
├── pf/
│   ├── consignado.md      # Crédito consignado — prazo, taxa mensal
│   └── veiculos.md        # Financiamento de veículos — entrada, prazo
└── pj/
    ├── capital_giro.md    # Capital de giro — prazo, taxa mensal
    └── recebiveis.md      # Antecipação de recebíveis — prazo, taxa
```

**Regra de escopo:** máximo 2–3 arquivos por pasta no MVP. Manter o contexto total por domínio abaixo de ~4.000 tokens para performance aceitável com Ollama local.

**Metadados futuros (Fase 3 — ChromaDB):** a estrutura de pastas já está preparada para ingestão com `DirectoryLoader` do LangChain, com metadados automáticos `{domain, topic, source}` derivados do caminho.

---

## 8. Estratégia de Testes

A estratégia de 100% mocks é mantida:

- `load_md_context` será mockado nos testes de nós — sem leitura real de disco.
- Novos testes unitários para `rag/loader.py` usando pasta temporária (`tmp_path` do pytest).
- Testes do router validam que `structured_query` é preenchido corretamente no estado.

---

## 9. Ambiente e Provider

| Ambiente | Provider LLM | Motivo |
|----------|-------------|--------|
| Desenvolvimento / testes | Ollama + `qwen2.5:7b` local (PC RTX 4060 via `192.168.0.200:11434`) | Zero custo, privacidade, funciona offline |
| Produção (cooperativa) | Provider externo — Groq + LLaMA 3, Claude Haiku ou similar | TTFT aceitável para atendimento em tempo real |

A arquitetura é **agnóstica de provider por design** — a troca é feita em uma linha no `.env` (`OLLAMA_BASE_URL`) ou instanciando um `ChatAnthropic`/`ChatGroq` no `build_graph`. Nenhum nó conhece o provider.

---

## 10. Arquivos Modificados

| Arquivo | Tipo | O que muda |
|--------|------|-----------|
| `state.py` | Modificado | Adiciona `structured_query` em `AgentState` e `RouterOutput` |
| `nodes/router.py` | Modificado | Novo system prompt com query rewriting; retorna `structured_query` |
| `nodes/agro.py` | Modificado | Remove `_AGRO_CONTEXT`; usa `load_md_context` + `structured_query` |
| `nodes/pf.py` | Modificado | Idem |
| `nodes/pj.py` | Modificado | Idem |
| `rag/loader.py` | Novo | `load_md_context(folder) -> str` |
| `rag/__init__.py` | Novo | Módulo vazio |
| `normativos/agro/*.md` | Novo | 2 arquivos de normativa rural |
| `normativos/pf/*.md` | Novo | 2 arquivos de normativa PF |
| `normativos/pj/*.md` | Novo | 2 arquivos de normativa PJ |
| `tests/test_loader.py` | Novo | Testes unitários do RAG loader |
| `tests/test_nodes.py` | Modificado | Atualiza mocks para `structured_query` |
| `tests/test_graph.py` | Modificado | Atualiza estado nos testes end-to-end |
| `requirements.txt` | Modificado | Adiciona `langchain-community>=0.3.0` |
| `ARCHITECTURE.md` | Modificado | Documenta Fase 2, query rewriting e loader |
