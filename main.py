from typing import TypedDict

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

WRITER_MODEL = "qwen2.5:7b-instruct"
JUDGE_MODEL = "qwen2.5:14b"


class FactCheck(BaseModel):
    claims: list[str] = Field(
        description="Each factual claim in the draft, marked SUPPORTED or UNSUPPORTED, quoting the research line that backs it."
    )
    fact_issues: list[str] = Field(
        description="Only claims that contradict the research or are absent from it. Empty list if every claim is supported.Only flag claims that contradict the research or are absent from it. Never comment on wording, tone, or style",
    )
    is_accurate: bool = Field(
        description="Assess whether the draft is accurate or not based on the claims and fact_issues.false if and only if issues is non-empty."
    )


class Critique(BaseModel):
    requirements_check: list[str] = Field(
        description="One line per explicit requirement in the assignment, each marked MET or NOT MET with a quote from the draft as evidence."
    )
    editor_feedback: list[str] = Field(
        description="Specific, actionable notes on what to improve. Do not rewrite the draftt. The complete critique itself, not an introduction to one. One entry per change needed. Empty list if the draft meets every requirement.Never state or reference the score here. This field contains only the list of changes to make.",
    )
    score: int = Field(
        ge=1,
        le=10,
        description="Score must be an integer from 1 to 10 based strictly on this tier system:\n"
        "- 10: Approve (Flawless / ready to use)\n"
        "- 8 to 9: Pass (Minor edits needed)\n"
        "- 2 to 6: Reject (Major edits / poor quality)\n"
        "- 1: Reject (Completely unusable)",
    )


writer_llm = ChatOllama(model=WRITER_MODEL, temperature=0.7, num_ctx=8192)
editor_llm = ChatOllama(
    model=WRITER_MODEL, temperature=0, num_ctx=8192
).with_structured_output(Critique)
fact_checker_llm = ChatOllama(
    model=JUDGE_MODEL, temperature=0, num_ctx=8192
).with_structured_output(FactCheck)
researcher_llm = ChatOllama(model=JUDGE_MODEL, temperature=0, num_ctx=8192)


class State(TypedDict, total=False):
    assignment: str
    current_draft: str
    editor_feedback: list[str]
    research: str
    score: int
    revision_count: int
    max_revisions: int
    fact_issues: list[str]
    is_accurate: bool


def writer_node(state: State) -> State:
    assignment = state.get("assignment", "")
    research = state.get("research", "")
    editor_feedback = state.get("editor_feedback", "")
    current_draft = state.get("current_draft", "")
    fact_issues = state.get("fact_issues", [])

    prompt_editor_feedback = (
        f"""Editor feedback: {editor_feedback}""" if editor_feedback else ""
    )
    prompt_fact_issues = (
        f"""Factual errors to fix{"\n".join(fact_issues)}""" if fact_issues else ""
    )
    draft_prompt = f"""Assignment: {assignment} Search results:{research}
   Write a text about the assingment using the search result.
   Return ONLY the draft text, no preamble
   Write in English, don't copy verbatim."""
    prompt = f"""Assignment: {assignment}   
    Information about the subject: {research}
    Your current draft is: {current_draft}. 
    {prompt_editor_feedback}
    {prompt_fact_issues}
    Return ONLY the draft text. No preamble, no explanation, no scores, no markdown headers.Write always in english.Use the research as your facts, but write the draft in your own words to satisfy the assignment. Do not copy the research verbatim.Change only what the listed issues identify. Leave everything else exactly as it is"""

    if not current_draft:
        new_draft = writer_llm.invoke(draft_prompt).content
    else:
        new_draft = writer_llm.invoke(prompt).content

    return {
        "current_draft": new_draft,
        "revision_count": state.get("revision_count", 0) + 1,
    }


def editor_node(state: State) -> State:
    assignment = state.get("assignment", "")
    current_draft = state.get("current_draft", "")

    draft_prompt = f"""Assignment: {assignment}
    Draft to review: {current_draft}

    Critique this draft against the assignment and score it.
    Judge the draft against the assignment.
    Penalise drafts that violate them. 
    Do not explain your score. 
    Do not include a rubric.
    List specific, actionable changes. 
    Each must name what to change and how. 
    No introduction, no summary.
    The assignment is the only standard. 
    If the draft satisfies it, score 8 or above. 
    Do not penalise a draft for being short if the assignment asked for short.
    The score must be consistent with your feedback. 
    Score 9–10 only if feedback is empty. If feedback is non-empty, score 8 or below
    If the draft fails any explicit requirement in the assignment, score 4 or below.
    Factual precision is the Fact-Checker's responsibility. Do not request claims the research does not support."."""

    critique = editor_llm.invoke(draft_prompt)

    return {"editor_feedback": critique.editor_feedback, "score": critique.score}


def researcher_node(state: State) -> State:
    assignment = state.get("assignment", "")
    try:
        search_results = DDGS().text(assignment, max_results=5)

    except DDGSException:
        print("An exception occurred")
        search_results = []

    research_prompt = f"""Assignment: {assignment} Search results:
    {search_results}
    
    Extract the key facts from these results. Use ONLY information present above.
    Do not add anything from your own knowledge.
    If sources disagree, state one value — the most specific and most frequently supported. Never present two conflicting dates."""
    research_result = researcher_llm.invoke(research_prompt)

    return {"research": research_result.content}


def fact_check_node(state: State) -> State:
    current_draft = state.get("current_draft", "")
    research = state.get("research", "")

    prompt = f"""Draft to fact-check: {current_draft}
Research (the only source of truth): {research}

List every factual claim from the draft in `claims`, each marked SUPPORTED or UNSUPPORTED against the research.
In `fact_issues`, list ONLY the UNSUPPORTED claims and what the research actually says instead.
If every claim is SUPPORTED, `fact_issues` must be an empty list and `is_accurate` must be true.
Never comment on wording, tone, or style.only judge verifiable claims — names, dates, titles, numbers — and ignore subjective characterisation.Only check verifiable specifics: names, dates, titles, numbers, quantities. Ignore subjective characterisation."""

    fact_check_result = fact_checker_llm.invoke(prompt)

    issues = [] if fact_check_result.is_accurate else fact_check_result.fact_issues

    return {
        "fact_issues": issues,
        "is_accurate": fact_check_result.is_accurate,
    }


def route_after_fact_check(state: State) -> str:
    if state.get("revision_count", 0) >= state.get("max_revisions", 3) or state.get(
        "is_accurate"
    ):
        return "proceed"
    else:
        return "revise"


def route_after_editor(state: State) -> str:
    if state.get("score", 0) >= 8:
        return "stop"
    if state.get("revision_count", 0) >= state.get("max_revisions", 3):
        return "stop"
    return "revise"


builder = StateGraph(State)
builder.add_node("writer", writer_node)
builder.add_node("editor", editor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("fact_checker", fact_check_node)
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "fact_checker")
builder.add_conditional_edges(
    "fact_checker", route_after_fact_check, {"revise": "writer", "proceed": "editor"}
)
builder.add_conditional_edges(
    "editor", route_after_editor, {"revise": "writer", "stop": END}
)
graph = builder.compile()


for event in graph.stream(
    {
        "assignment": "Explain in one sentence if god exists",
    }
):
    print("EVENT:")
    print(event)
    print("---")
