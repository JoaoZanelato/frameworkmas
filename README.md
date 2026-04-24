# FrameworkMAS

Sistema Multiagentes (MAS) baseado em grafos de estado para atendimento Tier 0 de suporte às agências de uma cooperativa de crédito. Desenvolvido com LangGraph e Ollama como projeto acadêmico (TCC — Análise e Desenvolvimento de Sistemas, previsão 2027).

---

## Sumário

1. [Problema e Contexto](#1-problema-e-contexto)
2. [Visão Geral da Arquitetura](#2-visão-geral-da-arquitetura)
3. [Stack Tecnológica](#3-stack-tecnológica)
4. [Estrutura de Arquivos](#4-estrutura-de-arquivos)
5. [Modelo de Estado](#5-modelo-de-estado)
6. [Os Nós do Grafo](#6-os-nós-do-grafo)
7. [RAG via Markdown](#7-rag-via-markdown)
8. [StubLLM — Testes sem LLM](#8-stubllm--testes-sem-llm)
9. [Instalação e Configuração](#9-instalação-e-configuração)
10. [Uso via CLI](#10-uso-via-cli)
11. [Testes](#11-testes)
12. [Como Adicionar Conteúdo Normativo](#12-como-adicionar-conteúdo-normativo)
13. [Como Adicionar um Novo Domínio](#13-como-adicionar-um-novo-domínio)
14. [Como Trocar o Provider de LLM](#14-como-trocar-o-provider-de-llm)
15. [Configuração do Ollama Remoto (GPU Windows)](#15-configuração-do-ollama-remoto-gpu-windows)
16. [Roadmap](#16-roadmap)

---

## 1. Problema e Contexto

### A Dor

O gerente de uma agência de cooperativa de crédito frequentemente precisa **pausar o atendimento com o associado** para ligar ao backoffice e tirar dúvidas sobre normativas de crédito: taxas do Pronaf, prazo do consignado, regras de capital de giro, condições de consórcio etc. Esse gargalo prejudica a experiência do associado e a produtividade da agência.

### A Solução

O FrameworkMAS é um **agente de IA local** que atua como Tier 0 de suporte. O gerente digita a dúvida em linguagem natural e recebe uma resposta técnica instantânea baseada nos normativos oficiais da cooperativa — sem pausar o atendimento, sem ligar para o backoffice e sem expor dados a serviços externos.

### Princípios de Design

- **Privacidade por design:** a arquitetura suporta inferência 100% local (Ollama em servidor interno) ou via API de mercado homologada — a decisão é operacional e não altera nenhum nó do sistema.
- **Precisão na primeira resposta:** o `router_node` reformula a pergunta bruta em uma query técnica completa antes de despachar para o especialista, eliminando a necessidade de interação extra.
- **Arquitetura modular:** cada domínio de conhecimento (AGRO, PF, PJ, CAMPANHAS) é um nó independente. Novos domínios são adicionados sem alterar os existentes.
- **Agnóstico a modelos:** a troca de Ollama para Groq, Claude ou qualquer outro provider é feita em uma linha, sem alterar nenhum nó.

---

## 2. Visão Geral da Arquitetura

O sistema é um **grafo de estado dirigido** implementado com LangGraph. Cada nó recebe o estado global, executa uma operação e devolve as chaves que modificou.

```
Pergunta do Gerente
        │
        ▼
┌───────────────────┐
│    router_node    │  ← classifica o domínio E reescreve a query (query rewriting)
│                   │    usa with_structured_output → RouterOutput
└────────┬──────────┘
         │  aresta condicional (lê state["domain"])
         │
         ├─── AGRO      ──► agro_node      ──► END
         ├─── PF        ──► pf_node        ──► END
         ├─── PJ        ──► pj_node        ──► END
         └─── CAMPANHAS ──► campanhas_node ──► END
```

### Fluxo do Estado

```
{ question }
    │
    ▼ router_node
{ question, domain, structured_query }
    │
    ▼ specialist_node (agro | pf | pj | campanhas)
{ question, domain, structured_query, context, generation }
```

Cada especialista recebe a `structured_query` (não o input bruto), carrega o contexto normativo dos arquivos `.md` do seu domínio e gera a resposta final.

---

## 3. Stack Tecnológica

| Componente | Tecnologia | Versão mínima | Motivo da escolha |
|---|---|---|---|
| Orquestração MAS | LangGraph | 1.0.0 | Grafos de estado determinísticos; roteamento condicional nativo; ideal para MAS com fluxos lineares |
| Abstração de LLM | LangChain | 1.0.0 | Abstrai o provider de LLM; troca de Ollama para qualquer API sem alterar os nós |
| LLM de desenvolvimento | Ollama + `qwen2.5:7b` | — | Execução local, zero custo, privacidade total dos normativos |
| Adapter Ollama | langchain-ollama | 1.1.0 | Integração LangChain ↔ Ollama com suporte a `with_structured_output` |
| Tipagem de estado | TypedDict (stdlib) | — | Contrato formal entre nós; exigido pelo LangGraph |
| Validação de saída | Pydantic v2 | 2.0.0 | Valida em runtime a saída estruturada do router; lança `ValidationError` se o LLM alucinar campos |
| Variáveis de ambiente | python-dotenv | 1.0.0 | Carrega `OLLAMA_BASE_URL` do `.env` sem hardcode |
| Testes | pytest + MagicMock | 8.0.0 | 100% isolados, sem chamadas reais ao LLM; runtime < 1s |
| Python | 3.11+ | — | Type hints modernos; `X \| Y` union syntax; compatível com todas as dependências |

### Motor de IA — Opções Suportadas

O sistema é agnóstico ao provider de LLM. Duas rotas são viáveis e a decisão depende de orçamento e política institucional:

| Opção | Provider | Custo recorrente | Privacidade | Prazo |
|---|---|---|---|---|
| **Infra local** | Ollama + GPU interna | Zero | Normativos 100% internos | Requer aquisição de hardware |
| **API de mercado** | Groq, Claude Haiku, OpenAI | Por uso (tokens) | Dados trafegam externamente | Pronto para uso imediato |

Em ambos os casos, a troca é feita em uma linha — nenhum nó especialista conhece o provider.

### Por que LangGraph e não um simples `if/else`?

O grafo de estado resolve três problemas que um `if/else` não resolve bem:

1. **Estado compartilhado e rastreável:** o `AgentState` é passado integralmente entre os nós; qualquer nó pode ler qualquer chave sem acoplamento direto.
2. **Extensibilidade declarativa:** adicionar um novo domínio é registrar um nó e uma aresta; o resto do grafo não muda.
3. **Auditoria nativa:** o LangGraph persiste o estado completo de cada passo, permitindo rastrear exatamente o que cada nó recebeu e devolveu.

---

## 4. Estrutura de Arquivos

```
FrameworkMAS/
│
├── state.py                        # AgentState TypedDict + RouterOutput Pydantic
├── main.py                         # Ponto de entrada CLI (argparse)
│
├── graphs/
│   ├── __init__.py
│   └── main_graph.py               # build_graph() + route_to_specialist()
│
├── nodes/
│   ├── __init__.py
│   ├── router.py                   # make_router_node() — classificação + query rewriting
│   ├── agro.py                     # make_agro_node() — especialista crédito rural
│   ├── pf.py                       # make_pf_node() — especialista crédito pessoal
│   ├── pj.py                       # make_pj_node() — especialista crédito empresarial
│   └── campanhas.py                # make_campanhas_node() — especialista campanhas
│
├── normativos/                     # Base de conhecimento em Markdown (RAG source)
│   ├── agro/
│   │   ├── pronaf.md               # Pronaf Mais Alimentos, Custeio, Mulher
│   │   ├── custeio_pecuario.md     # Bovinocultura, suinocultura
│   │   ├── investimento_rural.md   # Pronaf Investimento, Moderinfra, Finame Rural
│   │   └── seguro_rural.md         # Proagro, Proagro Mais, seguro agrícola privado
│   ├── pf/
│   │   ├── credito_pessoal.md      # Crédito livre, consignado, veículos, cartão
│   │   ├── habitacao.md            # Sicredi Casa, MCMV, reforma residencial
│   │   └── previdencia_seguro.md   # PGBL/VGBL, seguro de vida, prestamista, residencial
│   ├── pj/
│   │   ├── capital_giro.md         # Capital de giro, recebíveis, duplicatas
│   │   ├── investimento_pj.md      # BNDES Giro, Finame, modernização produtiva
│   │   └── conta_servicos_pj.md    # Conta PJ, Pix, ARV, maquininha
│   └── campanhas/
│       ├── movimentacao_do_bem.md  # Campanha de pontos por movimentação de conta
│       ├── capitalizacao.md        # Sicredi Cap — título de capitalização com sorteios
│       └── consorcio.md            # Consórcio Sicredi — imóveis, veículos, máquinas
│
├── stubs/
│   ├── __init__.py
│   └── stub_llm.py                 # StubLLM — LLM determinístico para testes offline
│
├── tests/
│   ├── test_state.py               # 3 testes: AgentState, RouterOutput, ValidationError
│   ├── test_nodes.py               # 15 testes: router (4) + especialistas (8) + campanhas (3)
│   └── test_graph.py               # 10 testes: route_to_specialist (5) + build_graph + end-to-end (4)
│
├── requirements.txt
├── pyproject.toml                  # Configuração do pytest (testpaths, pythonpath)
├── .env                            # OLLAMA_BASE_URL (não commitado)
├── ARCHITECTURE.md                 # Decisões arquiteturais detalhadas
└── CLAUDE.md                       # Diretrizes para o assistente de IA
```

---

## 5. Modelo de Estado

O estado global (`AgentState`) é um `TypedDict` que transita entre todos os nós do grafo. Cada nó recebe o estado completo e devolve apenas as chaves que modificou.

```python
# state.py
from typing import Literal, TypedDict
from pydantic import BaseModel


class AgentState(TypedDict):
    question: str          # Pergunta original do gerente — IMUTÁVEL (preservada para auditoria)
    structured_query: str  # Versão técnica e completa da pergunta, gerada pelo router_node
    domain: str            # Classificação: AGRO | PF | PJ | CAMPANHAS
    context: str           # Texto normativo carregado dos arquivos .md (RAG)
    generation: str        # Resposta final gerada pelo nó especialista


class RouterOutput(BaseModel):
    domain: Literal["AGRO", "PF", "PJ", "CAMPANHAS"]
    structured_query: str
```

### Por que `question` é imutável?

A pergunta original do gerente é preservada sem modificação do início ao fim. Isso garante:
- **Auditabilidade:** é possível rastrear exatamente o que o gerente perguntou vs. o que o sistema entendeu (`structured_query`).
- **Diagnóstico:** comparar `question` e `structured_query` revela se o query rewriting funcionou corretamente.

### Por que `RouterOutput` usa `Literal`?

O `Literal["AGRO", "PF", "PJ", "CAMPANHAS"]` no schema Pydantic instrui o LLM (via `with_structured_output`) a retornar **exatamente** um desses valores. Se o modelo alucinar um domínio inexistente, o Pydantic lança `ValidationError` em tempo de execução — tornando o problema visível e tratável.

---

## 6. Os Nós do Grafo

### 6.1 `router_node` — Classificador + Query Rewriting

**Arquivo:** `nodes/router.py`

O router tem **dupla responsabilidade** em uma única chamada ao LLM:

1. **Classificar** a pergunta em um dos 4 domínios disponíveis.
2. **Reescrever** a pergunta bruta em uma `structured_query` técnica e completa.

```python
def make_router_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    def router_node(state: AgentState) -> dict:
        structured_llm = llm.with_structured_output(RouterOutput)
        messages = [
            SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]
        result: RouterOutput = structured_llm.invoke(messages)
        return {
            "domain": result.domain,
            "structured_query": result.structured_query,
        }
    return router_node
```

**Por que query rewriting?**

Gerentes fazem perguntas curtas e ambíguas no calor do atendimento: `"e o trator?"`, `"carência?"`, `"qual o limite?"`. Sem reescrita, o especialista receberia um input insuficiente e poderia retornar uma resposta genérica ou incorreta.

Com query rewriting, essas perguntas se tornam:
- `"e o trator?"` → `"Qual a carência, taxa e prazo para financiamento de tratores via Pronaf Mais Alimentos para produtor familiar com DAP ativa?"`
- `"carência?"` → `"Qual o período de carência do Pronaf Custeio para safra de soja?"`

Isso garante **precisão na primeira chamada**, sem precisar de interação adicional com o gerente.

**Por que `with_structured_output`?**

Em vez de parsear texto livre com regex frágil, o LangChain instrui o modelo a retornar JSON válido conforme o schema `RouterOutput`. O resultado é determinístico e validado pelo Pydantic antes de chegar ao próximo nó.

---

### 6.2 Nós Especialistas — `agro_node`, `pf_node`, `pj_node`, `campanhas_node`

**Arquivos:** `nodes/agro.py`, `nodes/pf.py`, `nodes/pj.py`, `nodes/campanhas.py`

Todos os especialistas seguem o mesmo padrão — **factory function** que carrega o contexto normativo uma única vez no closure:

```python
def make_agro_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    context = _load_context()          # I/O ocorre aqui, UMA VEZ na inicialização

    def agro_node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=_AGRO_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=state["structured_query"]),  # usa a query reescrita
        ]
        response = llm.invoke(messages)
        return {
            "context": context,
            "generation": response.content,
        }

    return agro_node
```

**Pontos de design importantes:**

- O especialista recebe **`structured_query`**, não `question`. A pergunta já foi enriquecida pelo router.
- O contexto é carregado no `make_*_node()` e fechado no closure — **zero I/O por requisição**. Em produção com Ollama, o gargalo é a inferência, não a leitura de arquivo.
- O system prompt instrui o modelo a usar **somente o contexto fornecido** e a citar valores exatos. Isso reduz alucinação e ancora a resposta nos normativos reais.

| Nó | Pasta de normativos | Domínio |
|---|---|---|
| `agro_node` | `normativos/agro/` | Crédito rural, Pronaf, seguros rurais |
| `pf_node` | `normativos/pf/` | Crédito pessoal, habitação, previdência |
| `pj_node` | `normativos/pj/` | Capital de giro, investimento, conta PJ |
| `campanhas_node` | `normativos/campanhas/` | Campanhas, capitalização, consórcio |

---

### 6.3 `route_to_specialist` — Aresta Condicional

**Arquivo:** `graphs/main_graph.py`

Função pura (sem LLM) que lê `state["domain"]` e devolve o nome do próximo nó:

```python
def route_to_specialist(state: AgentState) -> Literal["agro_node", "pf_node", "pj_node", "campanhas_node"]:
    mapping = {
        "AGRO": "agro_node",
        "PF": "pf_node",
        "PJ": "pj_node",
        "CAMPANHAS": "campanhas_node",
    }
    return mapping.get(state["domain"], "pj_node")   # PJ como fallback seguro
```

Por ser uma função pura, é testada diretamente sem mock algum.

---

## 7. RAG via Markdown

### Como funciona

Cada nó especialista carrega automaticamente **todos os arquivos `.md`** da sua pasta de normativos ao ser instanciado:

```python
def _load_context() -> str:
    parts = []
    for md_file in sorted(_NORMATIVOS_DIR.glob("*.md")):
        parts.append(md_file.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)
```

Os arquivos são lidos em ordem alfabética, separados por `---`, e concatenados em uma única string que é injetada no system prompt do especialista.

### Vantagens desta abordagem para o MVP

| Característica | Benefício |
|---|---|
| Zero dependência externa | Não requer banco vetorial, embeddings ou servidor adicional |
| Zero I/O por requisição | Contexto carregado uma vez no `make_*_node()`; cada chamada usa a string em memória |
| Fácil manutenção | Equipe não-técnica pode editar os `.md` diretamente no repositório |
| Testabilidade | Testes verificam se o contexto contém palavras-chave esperadas (ex: `"Pronaf"` em AGRO) |

### Limitações e evolução planejada

O RAG via Markdown carrega **todo o conteúdo** de uma pasta para o contexto do LLM. Com muitos arquivos ou arquivos grandes, o contexto cresce e pode:
- Ultrapassar o limite de contexto do modelo
- Aumentar a latência e o custo por token

A evolução natural (Fase 3) é substituir o `_load_context()` por uma busca semântica via **ChromaDB ou FAISS**, onde apenas os trechos mais relevantes para a `structured_query` são recuperados. A interface dos nós especialistas não muda — apenas a implementação de `_load_context()`.

---

## 8. StubLLM — Testes sem LLM

**Arquivo:** `stubs/stub_llm.py`

O `StubLLM` é um LLM determinístico que não faz nenhuma chamada de rede. Ele classifica a pergunta por palavras-chave e retorna respostas template. É usado em dois contextos:

1. **Testes automatizados:** substituído por `MagicMock` nos testes unitários (controle total sobre o retorno).
2. **Desenvolvimento offline:** quando o Ollama não está disponível, `--stub` permite testar o pipeline completo localmente.

```python
class StubLLM(BaseChatModel):
    def _generate(self, messages, **kwargs) -> ChatResult:
        # retorna resposta template sem usar o contexto
        ...

    def with_structured_output(self, schema, **kwargs) -> _StructuredOutputWrapper:
        # wrapper que classifica por keywords e retorna RouterOutput
        ...
```

**Importante:** o StubLLM **não usa o contexto normativo** ao gerar respostas. Ele serve para validar que o pipeline (roteamento, carregamento de contexto, fluxo de estado) funciona — não para validar a qualidade das respostas.

**Palavras-chave por domínio:**

| Domínio | Exemplos de keywords |
|---|---|
| CAMPANHAS | `campanha`, `bem`, `movimentação`, `capitalização`, `consórcio`, `pontos`, `sorteio` |
| AGRO | `pronaf`, `trator`, `rural`, `safra`, `agrícola`, `proagro`, `lavoura` |
| PF | `consignado`, `veículo`, `cartão`, `habitação`, `previdência`, `inss` |
| PJ | fallback para qualquer outra pergunta |

> CAMPANHAS é verificado **antes** de AGRO/PF/PJ para evitar conflitos com palavras como "consórcio" (que poderia ser confundida com PJ).

---

## 9. Instalação e Configuração

### Pré-requisitos

- Python 3.11 ou superior
- Ollama instalado e rodando (local ou remoto) — [ollama.ai](https://ollama.ai)
- Modelo `qwen2.5:7b` baixado: `ollama pull qwen2.5:7b`

### Instalação

```bash
# Clone o repositório
git clone https://github.com/JoaoZanelato/frameworkmas.git
cd frameworkmas

# Instale as dependências
pip install -r requirements.txt
```

### Configuração do ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
# Ollama rodando localmente
OLLAMA_BASE_URL=http://localhost:11434

# Ollama rodando em outro host (ex: PC com GPU na rede local)
OLLAMA_BASE_URL=http://192.168.0.200:11434
```

Se o arquivo `.env` não existir, o sistema usa `http://localhost:11434` como padrão.

---

## 10. Uso via CLI

```bash
# Com Ollama (requer servidor Ollama ativo)
python main.py "Qual a carência do Pronaf Mais Alimentos para financiamento de trator?"

# Com pergunta sobre campanhas
python main.py "Como funciona a campanha movimentação do bem?"

# Sem LLM — modo offline determinístico (StubLLM)
python main.py --stub "Qual o prazo máximo do consignado?"

# Pergunta padrão (sem argumento)
python main.py
```

### Saída esperada

```
============================================================
PERGUNTA          : Qual a carência do Pronaf Mais Alimentos para financiamento de trator?
============================================================
DOMÍNIO           : AGRO

STRUCTURED QUERY  :
Qual o período de carência, taxa de juros e prazo total para financiamento
de tratores agrícolas via Pronaf Mais Alimentos para produtor familiar com DAP ativa?

CONTEXTO (RAG)    :
# Pronaf — Programa Nacional de Fortalecimento da Agricultura Familiar
...

------------------------------------------------------------
RESPOSTA FINAL:
Pelo Pronaf Mais Alimentos, o financiamento de tratores tem carência de 12 meses,
taxa de 5% ao ano e prazo total de até 10 anos.
============================================================
```

### Campos da saída

| Campo | Descrição |
|---|---|
| `PERGUNTA` | Input original do gerente (imutável) |
| `DOMÍNIO` | Classificação pelo router: AGRO, PF, PJ ou CAMPANHAS |
| `STRUCTURED QUERY` | Pergunta reescrita tecnicamente pelo router |
| `CONTEXTO (RAG)` | Conteúdo normativo carregado dos arquivos `.md` |
| `RESPOSTA FINAL` | Resposta gerada pelo especialista com base no contexto |

---

## 11. Testes

### Estratégia

100% dos testes são **isolados**: nenhum teste faz chamada real ao LLM ou ao Ollama. Os testes validam a lógica do grafo, o roteamento e o comportamento dos nós — não a qualidade das respostas do modelo.

```bash
# Rodar todos os testes
pytest -v

# Saída esperada: 23 passed in ~0.7s
```

### Cobertura atual

| Arquivo | Testes | O que valida |
|---|---|---|
| `test_state.py` | 3 | `AgentState` contém todas as chaves; `RouterOutput` valida campos; levanta `ValidationError` em campos faltantes |
| `test_nodes.py` | 15 | Router classifica os 4 domínios; especialistas usam `structured_query` (não `question`); contexto e geração são retornados |
| `test_graph.py` | 5 | `route_to_specialist` retorna o nó correto para cada domínio; domínio desconhecido cai em `pj_node`; `build_graph` compila sem erro; end-to-end para os 4 domínios |

### Padrão de mock dos testes

```python
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from state import RouterOutput

# Mock para o router (with_structured_output)
mock_llm = MagicMock()
mock_structured = MagicMock()
mock_structured.invoke.return_value = RouterOutput(
    domain="AGRO",
    structured_query="Qual a carência do Pronaf Mais Alimentos para tratores?",
)
mock_llm.with_structured_output.return_value = mock_structured

# Mock para os especialistas (invoke simples)
mock_llm.invoke.return_value = AIMessage(content="A carência é de 12 meses.")
```

---

## 12. Como Adicionar Conteúdo Normativo

Para enriquecer o conhecimento de um domínio existente, basta criar um novo arquivo `.md` na pasta correspondente. **Nenhuma mudança de código é necessária** — o `_load_context()` de cada especialista varre automaticamente todos os arquivos `.md` da pasta.

### Passo a passo

**1. Identifique o domínio correto:**

| Conteúdo | Pasta |
|---|---|
| Crédito rural, Pronaf, seguros agrícolas, maquinário rural | `normativos/agro/` |
| Crédito pessoal, habitação, previdência, seguro de vida | `normativos/pf/` |
| Capital de giro, investimento empresarial, conta PJ | `normativos/pj/` |
| Campanhas, capitalização, consórcio, pontos e prêmios | `normativos/campanhas/` |

**2. Crie o arquivo `.md` com um nome descritivo:**

```bash
# Exemplo: adicionar normativo de crédito consignado privado
touch normativos/pf/consignado_privado.md
```

**3. Estruture o conteúdo de forma clara:**

```markdown
# Crédito Consignado Privado

## Condições Gerais
- Taxa: a partir de 2,8% ao mês
- Prazo: até 84 meses
- Margem consignável: até 30% do salário líquido
- Convênios aceitos: empresas privadas com contrato ativo com a cooperativa

## Requisitos
- Vínculo empregatício ativo há no mínimo 6 meses
- Empresa conveniada na base da cooperativa
- Sem restrição cadastral no CPF

## Documentação
- Holerite dos últimos 3 meses
- Carteira de trabalho ou contrato de trabalho
- Autorização de desconto em folha assinada
```

**4. Verifique que o contexto foi carregado:**

```bash
python main.py --stub "Qual o prazo máximo do consignado privado?"
# O CONTEXTO (RAG) deve incluir o conteúdo do novo arquivo
```

**5. Rode os testes para garantir que nada quebrou:**

```bash
pytest -v
```

### Boas práticas para os arquivos `.md`

- **Use títulos e subtítulos** (`#`, `##`, `###`) para estruturar o conteúdo.
- **Use listas e tabelas** para taxas, prazos e condições — o LLM extrai valores numéricos com mais precisão de listas do que de parágrafos.
- **Cite valores exatos**: `5% ao ano`, `até R$ 165.000,00`, `prazo de 10 anos`.
- **Evite ambiguidade**: prefira `"limite de R$ 50.000,00 por beneficiário/ano-safra"` a `"limite alto"`.
- **Tamanho recomendado**: até ~150 linhas por arquivo. Arquivos muito grandes aumentam o contexto do LLM e podem degradar a qualidade da resposta.
- **Máximo recomendado por pasta (MVP)**: 3–5 arquivos para manter o contexto em tamanho gerenciável com Ollama local.

---

## 13. Como Adicionar um Novo Domínio

Siga os 6 passos abaixo para adicionar um novo domínio de A a Z. O exemplo usa o domínio `SEGUROS`.

### Passo 1 — Crie a pasta de normativos

```bash
mkdir normativos/seguros
touch normativos/seguros/seguro_auto.md
touch normativos/seguros/seguro_empresarial.md
```

### Passo 2 — Crie o nó especialista

Copie qualquer nó existente como base e ajuste 3 pontos:

```python
# nodes/seguros.py
from pathlib import Path
from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_NORMATIVOS_DIR = Path(__file__).parent.parent / "normativos" / "seguros"

_SEGUROS_SYSTEM_PROMPT = (
    "Você é um especialista em seguros da cooperativa Sicredi. "
    "Use SOMENTE o contexto normativo abaixo para responder ao gerente. "
    "Seja objetivo, cite os valores exatos e referencie o produto aplicável.\n\n"
    "Contexto:\n{context}"
)


def _load_context() -> str:
    parts = []
    for md_file in sorted(_NORMATIVOS_DIR.glob("*.md")):
        parts.append(md_file.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)


def make_seguros_node(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    context = _load_context()

    def seguros_node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=_SEGUROS_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=state["structured_query"]),
        ]
        response = llm.invoke(messages)
        return {
            "context": context,
            "generation": response.content,
        }

    return seguros_node
```

### Passo 3 — Atualize o `RouterOutput` em `state.py`

```python
# state.py
class RouterOutput(BaseModel):
    domain: Literal["AGRO", "PF", "PJ", "CAMPANHAS", "SEGUROS"]  # adicione aqui
    structured_query: str
```

Atualize também o comentário do campo `domain` em `AgentState`:

```python
domain: str  # Classificação: AGRO | PF | PJ | CAMPANHAS | SEGUROS
```

### Passo 4 — Atualize o system prompt do router em `nodes/router.py`

Adicione o novo domínio na lista de classificação:

```python
_ROUTER_SYSTEM_PROMPT = (
    ...
    "1. CLASSIFICAR em exatamente um domínio:\n"
    "   - AGRO: crédito rural, Pronaf, custeio agrícola/pecuário, máquinas rurais, seguro rural.\n"
    "   - PF: crédito pessoal, consignado, veículos, cartão de crédito, habitação, previdência.\n"
    "   - PJ: capital de giro, recebíveis, crédito empresarial, conta PJ, maquininha.\n"
    "   - CAMPANHAS: campanhas, movimentação do bem, capitalização, consórcio, pontos, sorteios.\n"
    "   - SEGUROS: seguro auto, seguro empresarial, sinistro, apólice, prêmio de seguro.\n"  # nova linha
    ...
)
```

### Passo 5 — Registre o nó e a aresta em `graphs/main_graph.py`

```python
from nodes.seguros import make_seguros_node  # import

def route_to_specialist(state):
    mapping = {
        "AGRO": "agro_node",
        "PF": "pf_node",
        "PJ": "pj_node",
        "CAMPANHAS": "campanhas_node",
        "SEGUROS": "seguros_node",  # adicione aqui
    }
    return mapping.get(state["domain"], "pj_node")

def build_graph(llm=None):
    ...
    builder.add_node("seguros_node", make_seguros_node(llm))  # registre o nó
    builder.add_edge("seguros_node", END)                      # e a aresta de saída
```

### Passo 6 — Atualize o `StubLLM` e adicione os testes

**Em `stubs/stub_llm.py`:**

```python
_SEGUROS_KEYWORDS = {"seguro auto", "apólice", "sinistro", "prêmio seguro", "seguro empresarial"}

def _classify(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in _CAMPANHAS_KEYWORDS): return "CAMPANHAS"
    if any(k in lower for k in _SEGUROS_KEYWORDS):   return "SEGUROS"   # adicione
    if any(k in lower for k in _AGRO_KEYWORDS):       return "AGRO"
    if any(k in lower for k in _PF_KEYWORDS):         return "PF"
    return "PJ"
```

**Em `tests/test_nodes.py`**, adicione dois testes seguindo o padrão dos existentes:

```python
def test_seguros_node_sets_context_and_generation(): ...
def test_seguros_node_uses_structured_query_not_raw_question(): ...
```

**Em `tests/test_graph.py`**, adicione:

```python
def test_route_to_specialist_returns_seguros_node():
    assert route_to_specialist(_state("SEGUROS")) == "seguros_node"

def test_graph_routes_seguros_question_end_to_end(): ...
```

---

## 14. Como Trocar o Provider de LLM

A troca de provider é feita em uma linha, sem alterar nenhum nó especialista. O LLM é injetado via `build_graph(llm=...)`.

### Trocar para Claude (Anthropic)

```python
# Em graphs/main_graph.py ou em um script externo
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.0)
graph = build_graph(llm=llm)
```

Instale: `pip install langchain-anthropic` e defina `ANTHROPIC_API_KEY` no `.env`.

### Trocar para Groq (alta velocidade, TTFT baixo)

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
graph = build_graph(llm=llm)
```

Instale: `pip install langchain-groq` e defina `GROQ_API_KEY` no `.env`.

### Trocar para OpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
graph = build_graph(llm=llm)
```

Instale: `pip install langchain-openai` e defina `OPENAI_API_KEY` no `.env`.

### Manter Ollama com outro modelo

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434", temperature=0.0)
graph = build_graph(llm=llm)
```

> **Requisito crítico:** o provider escolhido precisa suportar `with_structured_output()` do LangChain. Todos os listados acima suportam. Para providers menos comuns, verifique a documentação do langchain correspondente.

---

## 15. Configuração do Ollama Remoto (GPU Windows)

Se o notebook de desenvolvimento não tem GPU mas há um PC com placa dedicada na rede local, é possível rodar o Ollama no PC e acessar remotamente pelo notebook.

### No PC com GPU (Windows)

**1. Instale o Ollama:** [ollama.ai/download](https://ollama.ai/download)

**2. Baixe o modelo:**
```powershell
ollama pull qwen2.5:7b
```

**3. Configure o Ollama para aceitar conexões externas:**

Abra "Editar variáveis de ambiente do sistema" e crie:
- Nome: `OLLAMA_HOST`
- Valor: `0.0.0.0`

**4. Reinicie o Ollama** (feche e abra pelo ícone na bandeja do sistema).

**5. Libere a porta no firewall (PowerShell como Administrador):**
```powershell
New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
```

### No notebook (Linux/Mac)

```bash
# Descubra o IP do PC na rede local
# Windows: ipconfig | No terminal Linux: nmap -sn 192.168.0.0/24

# Configure o .env
echo "OLLAMA_BASE_URL=http://192.168.0.200:11434" > .env

# Teste a conectividade
curl http://192.168.0.200:11434/api/tags
```

---

## 16. Roadmap

### ✅ Fase 3 — Domínio CAMPANHAS (concluída)

- Quarto nó especialista: campanhas sazonais, capitalização, consórcio
- Base normativa: `normativos/campanhas/`
- 23/23 testes passando

### Fase 4 — Vector Store

Substituir o RAG via Markdown por busca semântica real. Apenas `_load_context()` muda — a interface dos nós permanece idêntica.

| Item | Descrição |
|---|---|
| ChromaDB ou FAISS | Banco vetorial local para embeddings dos normativos |
| Embeddings | `nomic-embed-text` via Ollama (zero custo, local) ou `text-embedding-3-small` da OpenAI |
| Chunking | Dividir os `.md` em chunks de ~500 tokens com overlap de 50 tokens |
| Retrieval | Top-k (k=3–5) chunks mais similares à `structured_query` |
| Reindexação | Script para reindexar automaticamente ao detectar mudanças nos `.md` |

### Fase 5 — API HTTP (DevConsole)

Expor o motor de inferência via FastAPI para integração com o DevConsole da Sicredi.

```
POST /query
Body: { "question": "Qual a carência do Pronaf?" }
Response: { "domain": "AGRO", "structured_query": "...", "generation": "..." }
```

### Fase 6 — Deploy em Produção

| Item | Descrição |
|---|---|
| Servidor | `cas` / `devconsole` interno da cooperativa |
| LLM | Infra local (Ollama com GPU interna) ou API de mercado homologada — a definir |
| Persistência | Histórico de conversas por agência para auditoria e melhoria contínua |
| Monitoramento | Logging estruturado de `question`, `domain`, `structured_query` e latência |

---

## Contribuindo

1. Crie uma branch a partir de `main`: `git checkout -b feat/novo-dominio`
2. Implemente seguindo as seções [12](#12-como-adicionar-conteúdo-normativo) ou [13](#13-como-adicionar-um-novo-domínio)
3. Garanta que todos os testes passam: `pytest -v`
4. Abra um Pull Request com descrição do domínio adicionado e exemplos de perguntas atendidas

---

*FrameworkMAS — Trabalho de Conclusão de Curso em Análise e Desenvolvimento de Sistemas. João Zanelato, 2027.*
