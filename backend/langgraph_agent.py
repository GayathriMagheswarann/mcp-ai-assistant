
import asyncio
import re
import time
from typing import Annotated
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from mcp_tools import search_files, read_file, search_file_content


# --------------------------------------------------
# Agent State
# --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --------------------------------------------------
# Local LLM
# --------------------------------------------------

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    num_predict=700,
    reasoning=False,
)

study_llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    num_predict=350,
    reasoning=False,
)



# --------------------------------------------------
# MCP Tools
# --------------------------------------------------

tools = [
    search_files,
    read_file,
    search_file_content,
]

llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# Python Request Router
# --------------------------------------------------

def route_request(user_input: str) -> str:
    text = user_input.lower().strip()

    # A named PDF or TXT file means search within that file.
    if re.search(
        r"\b[\w.-]+(?:\s+[\w.-]+)*\.(?:pdf|txt)\b",
        text,
        re.IGNORECASE,
    ):
        return "search_file_content"

    # Requests to locate files.
    file_search_phrases = (
        "find my",
        "find the notes",
        "find the files",
        "list my files",
        "show my files",
        "search for files",
        "unit 1 notes",
        "unit 2 notes",
    )

    if any(phrase in text for phrase in file_search_phrases):
        return "search_files"

    # Explicit requests to use study materials.
    study_phrases = (
        "from my notes",
        "from my unit",
        "in my notes",
        "in my unit",
        "from the pdf",
        "in the pdf",
        "from my pdf",
        "in my study material",
        "from my study material",
    )

    if any(phrase in text for phrase in study_phrases):
        return "search_file_content"

    # Common study questions search available notes.
    question_starters = (
        "what is ",
        "what are ",
        "explain ",
        "define ",
        "describe ",
        "discuss ",
        "tell me about ",
    )

    if text.startswith(question_starters):
        return "search_file_content"

    # Everything else goes through the general LLM agent.
    return "llm"


# --------------------------------------------------
# Helper: Clean and Validate Model Output
# --------------------------------------------------

def clean_answer(content) -> str:
    """Clean known reasoning markers and return readable text."""

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))

        content = "\n".join(parts)

    if not isinstance(content, str):
        content = str(content)

    content = content.strip()

    # Remove completed <think>...</think> sections.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()

    # If a reasoning section was opened but never closed,
    # do not display the unfinished reasoning.
    elif "<think>" in content:
        content = content.split("<think>", 1)[0].strip()

    # Remove a leading /no_think directive if the model echoes it.
    content = re.sub(
        r"^\s*/no_think\s*",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()

    # Detect common reasoning-style openings.
    # This is a safety check, not a complete reasoning detector.
    reasoning_start = re.compile(
        r"^\s*(?:"
        r"hmm,\s+the user\b|"
        r"the user (?:wants|asks|is asking|specifically)\b|"
        r"let me (?:scan|check|think|analyze|craft|look)\b|"
        r"i should (?:focus|make sure|include|provide|extract)\b|"
        r"i need to (?:give|provide|answer|explain|check)\b|"
        r"i'll (?:make sure|focus|craft|extract|provide)\b|"
        r"we need to (?:answer|explain|focus|check)\b|"
        r"final answer should\b|"
        r"let's scan through\b"
        r")",
        re.IGNORECASE,
    )

    if reasoning_start.search(content):
        return ""

    return content.strip()


def generation_was_truncated(response) -> bool:
    """Check whether Ollama reports that generation hit its limit."""

    metadata = getattr(response, "response_metadata", {}) or {}
    return str(metadata.get("done_reason", "")).lower() == "length"


def display_answer(response) -> None:
    """Display a final answer and flag incomplete generation."""

    answer = clean_answer(response.content)

    if answer:
        print("\nAI:", answer)
    else:
        print(
            "\nAI: The model did not return a clean final answer. "
            "Please try the question again."
        )

    if generation_was_truncated(response):
        print(
            "\n[WARNING] Generation reached its output limit. "
            "The answer may be incomplete."
        )


# --------------------------------------------------
# Assistant Node
# --------------------------------------------------

async def assistant_node(state: AgentState):
    start_time = time.perf_counter()

    response = await llm_with_tools.ainvoke(
        [
            (
                "system",
                "/no_think\n"
                "You are a tool-using AI assistant. "
                "When the user asks you to search for files, you MUST call the search_files tool. "
                "When the user asks to read a file, you MUST call read_file. "
                "When the user asks to find information inside notes, you MUST call search_file_content."
                "Never describe a tool call in text instead of making the actual call. "
                "After receiving tool results, answer using those results. "
                "Do not narrate your reasoning. "
                "Return only the final answer.",
            ),
            *state["messages"],
        ]
    )

    elapsed = time.perf_counter() - start_time

    print(
        f"\n[DEBUG] Qwen3 response time: {elapsed:.2f} seconds"
    )
    print(
        f"[DEBUG] Tool calls requested: {len(response.tool_calls)}"
    )

    return {"messages": [response]}


# --------------------------------------------------
# Decide Whether to Continue
# --------------------------------------------------

def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

builder.add_edge("tools", "assistant")

graph = builder.compile()


# --------------------------------------------------
# Extract Filename and Search Topic
# --------------------------------------------------

def extract_search_details(user_input: str):
    """Extract an optional filename and the actual search topic."""

    filename = ""

    # Supports names such as MPC-UNIT 1.pdf and PYTHON INTRO.txt.
    filename_match = re.search(
        r"\bfrom\s+my\s+(.+?\.(?:pdf|txt))\b",
        user_input,
        re.IGNORECASE,
    )

    if not filename_match:
        filename_match = re.search(
            r"\b([\w.-]+(?:\s+[\w.-]+)*\.(?:pdf|txt))\b",
            user_input,
            re.IGNORECASE,
        )

    topic = user_input

    if filename_match:
        filename = filename_match.group(1).strip()
        topic = re.sub(
            re.escape(filename),
            "",
            topic,
            flags=re.IGNORECASE,
        )

    topic = re.sub(
        r"\b(from|in)\s+(my|the)\b",
        "",
        topic,
        flags=re.IGNORECASE,
    )

    topic = re.sub(
        r"^(explain|describe|define|discuss|tell me about|what is|what are)\s+",
        "",
        topic,
        flags=re.IGNORECASE,
    )

    topic = re.sub(r"[?!.]+$", "", topic).strip()
    topic = re.sub(r"\s+", " ", topic)

    return topic or user_input, filename


# --------------------------------------------------
# Interactive Application
# --------------------------------------------------

async def main():
    print("MCP-Powered AI Assistant")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("AI: Goodbye!")
            break

        if not user_input:
            continue

        total_start = time.perf_counter()
        print("\nProcessing your request...")

        try:
            route = route_request(user_input)
            print(f"[DEBUG] Selected route: {route}")

            # ------------------------------------------
            # Search for files
            # ------------------------------------------

            if route == "search_files":
                retrieval_start = time.perf_counter()

                result = await search_files.ainvoke(
                    {"query": user_input}
                )

                retrieval_time = (
                    time.perf_counter() - retrieval_start
                )

                print(
                    f"[DEBUG] MCP file-search time: "
                    f"{retrieval_time:.2f} seconds"
                )

                print("\nAI: Matching files:")
                print(result)

            # ------------------------------------------
            # Search study materials
            # ------------------------------------------

            elif route == "search_file_content":
                topic, filename = extract_search_details(
                    user_input
                )

                print(f"[DEBUG] Search topic: {topic}")
                print(
                    "[DEBUG] PDF/TXT filename: "
                    f"{filename or '(search all files)'}"
                )

                # Measure MCP retrieval separately.
                retrieval_start = time.perf_counter()

                result = await search_file_content.ainvoke(
                    {
                        "query": topic,
                        "filename": filename,
                    }
                )

                retrieval_time = (
                    time.perf_counter() - retrieval_start
                )

                print(
                    f"[DEBUG] MCP retrieval time: "
                    f"{retrieval_time:.2f} seconds"
                )

                result_text = str(result)
                result_lower = result_text.lower()

                # Handle missing or inaccessible named files.
                if filename and (
                    "file not found" in result_lower
                    or "access denied" in result_lower
                ):
                    print("\nAI:", result_text)

                # Do not generate an answer from unrelated passages.
                elif (
                    not result
                    or "no matching content found" in result_lower
                ):
                    print(
                        "\nAI: I couldn't find relevant information "
                        "in your available notes. You can ask me to "
                        "answer using general knowledge instead."
                    )

                else:
                    # Measure Qwen3 generation separately.
                    generation_start = time.perf_counter()

                    response = await llm.ainvoke(
                        [
                            (
                                "system",
                                "/no_think\n"
                                "You are an exam study assistant. "
                                "Answer the question directly using only "
                                "the retrieved passages. "
                                "Do not invent details. "
                                "Do not describe your thinking process "
                                "or narrate your analysis. "
                                "Start with the definition or main answer. "
                                "Use short bullet points when useful. "
                                "If the passages are insufficient, say so. "
                                "Include the key facts needed to answer the question. "
                                "Finish the explanation completely. "
                                "If the passages are insufficient, say so. "
                                "Return only the final answer.",
                            ),
                            HumanMessage(
                                content=(
                                    "/no_think\n"
                                    "Give a concise, complete answer. "
                                    "Avoid repetition and unnecessary introductions.\n\n"
                                    f"Question: {user_input}\n\n"
                                    f"Retrieved passages:\n{result_text}\n\n"
                                    "Answer using these passages."
                                )
                            ),
                        ],
                        config={"recursion_limit": 10},

                    )
                    metadata = response.response_metadata or {}

                    eval_count = metadata.get("eval_count")
                    eval_duration = metadata.get("eval_duration")
                    prompt_eval_count = metadata.get("prompt_eval_count")
                    prompt_eval_duration = metadata.get("prompt_eval_duration")

                    print("\n[DEBUG] Qwen3 performance metrics:")

                    if eval_count is not None:
                        print(f"[DEBUG] Generated tokens: {eval_count}")

                    if eval_duration:
                        generation_seconds = eval_duration / 1_000_000_000
                        tokens_per_second = eval_count / generation_seconds if eval_count else 0

                        print(
                            f"[DEBUG] Generation duration: "
                            f"{generation_seconds:.2f} seconds"
                        )
                        print(
                            f"[DEBUG] Generation speed: "
                            f"{tokens_per_second:.2f} tokens/second"
                        )
                    else:
                        print("[DEBUG] Generation duration: unavailable")

                    if prompt_eval_count is not None:
                        print(f"[DEBUG] Prompt tokens: {prompt_eval_count}")

                    if prompt_eval_duration:
                        prompt_seconds = prompt_eval_duration / 1_000_000_000
                        print(
                            f"[DEBUG] Prompt processing: "
                            f"{prompt_seconds:.2f} seconds"
                        )


                    generation_time = (
                        time.perf_counter() - generation_start
                    )

                    print(
                        f"[DEBUG] Qwen3 generation time: "
                        f"{generation_time:.2f} seconds"
                    )

                    display_answer(response)

                    metadata = response.response_metadata or {}
                    if metadata:
                        print(
                            "\n[DEBUG] Generation finished: "
                            f"{metadata.get('done_reason', 'unknown')}"
                        )

            # ------------------------------------------
            # General conversation through LangGraph
            # ------------------------------------------

            else:
                result = await graph.ainvoke(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ]
                    }
                )

                final_message = result["messages"][-1]

                # Display the final response through the same cleaner.
                display_answer(final_message)

        except Exception as error:
            print(f"\nError: {error}")

        finally:
            total_time = time.perf_counter() - total_start
            print(
                f"\n[DEBUG] Total request time: "
                f"{total_time:.2f} seconds"
            )


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
