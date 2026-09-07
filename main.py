from typing import TypedDict

from ddgs import DDGS
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import NotRequired


class Critique(BaseModel):
    feedback: str = Field(
        min_length=2,
        description="Specific, actionable notes on what to improve. Do not rewrite the draftt. The complete critique itself, not an introduction to one. 2–4 specific bullet points.Never state or reference the score here. This field contains only the list of changes to make.",
    )
    score: int = Field(
        ge=1,
        le=10,
        description="Score must be an integer from 1 to 10 based strictly on this tier system:\n"
        "- 9 to 10: Approve (Flawless / ready to use)\n"
        "- 7 to 8: Pass (Minor edits needed)\n"
        "- 2 to 6: Reject (Major edits / poor quality)\n"
        "- 1: Reject (Completely unusable)",
    )


writer_llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.7, num_ctx=8192)
editor_llm = ChatOllama(
    model="qwen2.5:7b-instruct", temperature=0, num_ctx=8192
).with_structured_output(Critique)
researcher_llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, num_ctx=8192)


class State(TypedDict, total=False):
    assignment: str
    current_draft: str
    feedback: str
    research: str
    score: int
    count: int
    ceiling: int


def write_node(state: State):
    assignment = state.get("assignment", "")
    research = state.get("research", "")
    feedback = state.get("feedback", "")
    current_draft = state.get("current_draft", "")

    output = ""
    first_prompt = f"""Assignment: {assignment} Search results:{research}
   Write a text about the assingment using the search result."""
    aux_prompt = f"""Assignment: {assignment}
    Information about the subject: {research}
    Your current draft: {current_draft}
    Editor feedback: {feedback}
    Rewrite the draft to address the feedback
    Return ONLY the draft text. No preamble, no explanation, no scores, no markdown headers.Write always in english"""
    if not feedback:
        output = writer_llm.invoke(first_prompt).content
    else:
        output = writer_llm.invoke(aux_prompt).content

    return {"current_draft": output, "count": state.get("count", 0) + 1}


def editor_node(state: State) -> dict:
    assignment = state.get("assignment", "")
    current_draft = state.get("current_draft", "")

    aux_prompt = f"""Assignment: {assignment}
    Draft to review: {current_draft}

    Critique this draft against the assignment and score it.
    Judge the draft against the assignment, including any length or format constraints. Penalise drafts that violate them. Do not explain your score. Do not include a rubric.List 2–4 specific, actionable changes. Each must name what to change and how. No introduction, no summary.The assignment is the only standard. If the draft satisfies it, score 8 or above. Do not penalise a draft for being short if the assignment asked for short.The score must be consistent with your feedback. If you list problems, do not score above 8.You must list at least two concrete changes, or state 'No changes needed."""

    critique = editor_llm.invoke(aux_prompt)

    return {"feedback": critique.feedback, "score": critique.score}


def research_node(state: State) -> dict:
    assignment = state.get("assignment", "")
    try:
        results = DDGS().text(assignment, max_results=5)
    except NameError:
        print("An exception occurred")
        results = []
    aux_prompt = f"""Assignment: {assignment} Search results:
    {results}
    
    Extract the key facts from these results. Use ONLY information present above.
    Do not add anything from your own knowledge."""
    research_result = researcher_llm.invoke(aux_prompt)

    return {"research": research_result.content}


def should_continue(state: State) -> str:
    if state.get("score", 0) >= 8:
        return "stop"
    if state.get("count", 0) >= state.get("ceiling", 3):
        return "stop"
    return "revise"


builder = StateGraph(State)
builder.add_node("writer", write_node)
builder.add_node("editor", editor_node)
builder.add_node("researcher", research_node)
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "editor")
builder.add_conditional_edges(
    "editor", should_continue, {"revise": "writer", "stop": END}
)
graph = builder.compile()

for event in graph.stream(
    {
        "assignment": "Write one sentence about Pepe the Frog. It must mention the year and the artist.",
        "ceiling": 3,
    }
):
    print("EVENT:")
    print(event)
    print("---")
