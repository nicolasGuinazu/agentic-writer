# agentic-writer | LangGraph multi-agent workflow

A four-agent writing pipeline built with [LangGraph](https://langchain-ai.github.io/langgraph/),
running entirely on local models via [Ollama](https://ollama.com). No API keys, no cloud calls.

Give it a writing assignment. It researches the topic on the web, drafts a text,
fact-checks the draft against what it found, edits for quality, and loops until the
draft passes both checks or runs out of revisions.

## Purpose

This is a learning project. The goal was to understand **agent orchestration** from the
ground up rather than through a framework's high-level abstractions: how independent
agents share state, how conditional routing creates loops, how you guarantee a cyclic
graph terminates, and how you get structured, machine-readable output out of a language
model instead of prose you have to parse.

It is deliberately built from primitives — `StateGraph`, plain node functions and conditional
edges.

## Architecture


START
│
▼
researcher ──► writer ──► fact_checker ──revise──► writer
│
proceed
│
▼
editor ──revise──► writer
│
stop
│
▼
END


**Two independent revision cycles**, each with its own termination guard:

| Agent | Owns | Job |
|---|---|---|
| `researcher` | `research` | Web-searches the topic and extracts facts from the results only |
| `writer` | `current_draft`, `revision_count` | Drafts, then revises using whatever critique exists |
| `fact_checker` | `fact_issues`, `is_accurate` | Checks every claim in the draft against the research |
| `editor` | `editor_feedback`, `score` | Judges the draft against the assignment and scores it 1–10 |

Agents never call each other. They communicate only by reading from and writing to a
shared state dictionary, which is what makes the loops and branches possible.

## Requirements

- Python 3.13
- [Ollama](https://ollama.com) running locally
- ~16 GB RAM (the judge model is ~9 GB)
- Tested on macOS / Apple Silicon

## Install

```bash
git clone <your-repo-url>
cd <repo>

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Install Ollama, then pull the models:

```bash
ollama pull qwen2.5:7b-instruct   # writer + researcher
ollama pull qwen2.5:14b           # fact-checker + editor
```

Make sure the Ollama server is running (launch the app, or `ollama serve`) — the graph
talks to it on `localhost:11434`.

## Usage

```bash
python main.py
```

Edit the assignment at the bottom of the file:

```python
for event in graph.stream({
    "assignment": "Write one sentence about Pepe the Frog. It must mention the year and the artist.",
    "max_revisions": 3,
}):
    print(event)
```

`.stream()` prints one event per node so you can watch each agent fire and see exactly
which keys it wrote. Use `.invoke()` instead if you only want the final state.

## Design notes

**Two models, chosen on purpose.** The two agents that must fill a strict schema and
exercise judgment run on the larger model; drafting prose runs on the smaller, faster one.
Nearly every failure during development came from a judge misfiring, not from the writer.

**Structured output, with real constraints.** The judges return Pydantic models rather than
text, so their verdicts are typed values the router can branch on. Constraints like
`ge=1, le=10` on the score are load-bearing: a prompt asking for 1–10 was ignored, the
schema constraint was not.

**Every cycle has a guard.** `max_revisions` bounds both loops. Two independent exit
conditions — quality threshold and hard count — because an LLM judge alone can't be
trusted to ever say "good enough."

**Empty states must be legal.** A field with no valid way to express "nothing here" gets
filled with noise — invented critiques, the string `"None"`, restated draft text. Lists
that can be empty, plus a rule tying the verdict to emptiness, removed a whole class of bug.

**Normalise at the node boundary.** Schemas make good output likely; a line of Python makes
bad output impossible. Nodes are the last place you control a value before it enters
shared state.

## Known limitations

**The fact-checker verifies provenance, not truth.** It answers "is this claim traceable to
the research?", never "is this claim true?" A well-grounded claim from a wrong source still
passes. Overall accuracy is capped by retrieval quality, and no prompt changes that.

**The researcher runs once.** If the fact-checker finds a claim *absent* from the research
(rather than contradicting it), the only available repair is to send the writer back — which
can't work, because the missing information isn't in state. The graph needs a
`fact_checker → researcher` edge to handle that properly.

**The search query is the raw assignment.** A writing instruction makes a poor search query,
so retrieval quality varies between runs. Deriving a short query from the assignment would fix it.

**Strict fact-checking flattens the prose.** Because the research is the only permitted source
of truth, the writer converges toward paraphrasing it. Accurate, but encyclopaedic.

**Judges are noisy.** Even at `temperature=0`, near-identical drafts can score differently
between runs. The revision ceiling exists partly to contain this.

## Roadmap

- `fact_checker → researcher` edge, distinguishing "contradicts research" from "absent from research"
- Run the fact-checker and editor in parallel instead of in series (fan-out / fan-in)
- A checkpointer, so state survives between invocations
- `interrupt()` for human approval before a draft is accepted
- Tool-calling, so the researcher decides what to search rather than always searching the assignment