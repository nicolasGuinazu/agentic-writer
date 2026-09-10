from typing import TypedDict

from ddgs import DDGS
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import NotRequired


class FactCheck(BaseModel):
    claims: list[str] = Field(
        description="Each factual claim in the draft, marked SUPPORTED or UNSUPPORTED, quoting the research line that backs it."
    )
    issues: list[str] = Field(
        description="Any factual issues found in the draft", min_length=1
    )
    is_accurate: bool = Field(
        description="Access if the draft is accurate or not based on the claims and issues"
    )


class Critique(BaseModel):
    requirements_check: list[str] = Field(
        description="One line per explicit requirement in the assignment, each marked MET or NOT MET with a quote from the draft as evidence."
    )
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
fact_checker_llm = ChatOllama(
    model="qwen2.5:7b-instruct", temperature=0, num_ctx=8192
).with_structured_output(FactCheck)
researcher_llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, num_ctx=8192)


class State(TypedDict, total=False):
    assignment: str
    current_draft: str
    feedback: str
    research: str
    score: int
    count: int
    ceiling: int
    issues: str
    is_accurate: bool


def write_node(state: State) -> State:
    assignment = state.get("assignment", "")
    research = state.get("research", "")
    feedback = state.get("feedback", "")
    current_draft = state.get("current_draft", "")
    issues = state.get("issues", "")

    output = ""
    first_prompt = f"""Assignment: {assignment} Search results:{research}
   Write a text about the assingment using the search result.
   Write in English, don't copy verbatim."""
    aux_prompt = f"""Assignment: {assignment}
    Information about the subject: {research}
    If no draft yet, write one {current_draft}. If the draft exists → revise it using the Editor feedback: {feedback}
    Return ONLY the draft text. No preamble, no explanation, no scores, no markdown headers.Write always in english.Use the research as your facts, but write the draft in your own words to satisfy the assignment. Do not copy the research verbatim.
    Factual errors to fix{"\n".join(issues)}"""
    if not feedback:
        output = writer_llm.invoke(first_prompt).content
    else:
        output = writer_llm.invoke(aux_prompt).content

    return {"current_draft": output, "count": state.get("count", 0) + 1}


def editor_node(state: State) -> State:
    assignment = state.get("assignment", "")
    current_draft = state.get("current_draft", "")

    aux_prompt = f"""Assignment: {assignment}
    Draft to review: {current_draft}

    Critique this draft against the assignment and score it.
    Judge the draft against the assignment, including any length or formneededat constraints. Penalise drafts that violate them. Do not explain your score. Do not include a rubric.List 2–4 specific, actionable changes. Each must name what to change and how. No introduction, no summary.The assignment is the only standard. If the draft satisfies it, score 8 or above. Do not penalise a draft for being short if the assignment asked for short.The score must be consistent with your feedback. If you list problems, do not score above 8If the draft fails any explicit requirement in the assignment, score 4 or below."."""

    critique = editor_llm.invoke(aux_prompt)

    return {"feedback": critique.feedback, "score": critique.score}


def research_node(state: State) -> State:
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


def fact_check_node(state: State) -> State:
    assignment = state.get("assignment", "")
    current_draft = state.get("current_draft", "")
    research = state.get("research", "")

    aux_prompt = f"""Assignment: {assignment}
    Draft to fact-check: {current_draft}
    Research information: {research}

    Fact-check the draft against all the claims
    
    Identify every single one and determine if they are supported or unsupported by the research """

    fact_check_result = fact_checker_llm.invoke(aux_prompt)
    print(fact_check_result.claims)
    return {
        "issues": fact_check_result.issues,
        "is_accurate": fact_check_result.is_accurate,
    }


def should_continue_to_editor(state: State) -> str:
    if state.get("count", 0) >= state.get("ceiling", 3) or state.get("is_accurate"):
        return "proceed"
    else:
        return "revise"


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
builder.add_node("fact_checker", fact_check_node)
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "fact_checker")
builder.add_conditional_edges(
    "fact_checker", should_continue_to_editor, {"revise": "writer", "proceed": "editor"}
)
builder.add_conditional_edges(
    "editor", should_continue, {"revise": "writer", "stop": END}
)
graph = builder.compile()


# print(
#     fact_check_node(
#         {
#             "assignment": "Write one sentence about Pepe the Frog. It must mention the year and the artist.",
#             "current_draft": "Pepe is a yellow frog, created in 21900 by Jordan Peterson.",
#         }
#     )
# )
for event in graph.stream(
    {
        "assignment": "Write one sentence about Pepe the Frog. It must mention the year and the artist.",
        "current_draft": "Pepe is a yellow frog, created in 21900 by Jordan Peterson.",
        "ceiling": 3,
    }
):
    print("EVENT:")
    print(event)
    print("---")
