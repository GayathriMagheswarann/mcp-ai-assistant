
import asyncio
from urllib import response
from langchain_ollama import ChatOllama
from langchain_core.tools import tool


@tool
def search_file_content(query: str) -> str:
    """Search study notes for a topic and return relevant content."""
    return (
        "From Unit 1 notes: DSSS means Direct Sequence Spread Spectrum. "
        "The data bit stream is combined with a chipping sequence to spread "
        "the signal over a wider bandwidth."
    )


async def main():
    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        num_predict=600,
    )

    model_with_tools = llm.bind_tools(
        [search_file_content],
        tool_choice="required",
    )

    response = await model_with_tools.ainvoke(
        "Search the notes for DSSS before answering."
    )

    print("Tool calls:", response.tool_calls)
    print("Additional kwargs:", response.additional_kwargs)
    print("Raw response:", response)
    print("Response metadata:", response.response_metadata)
    print("Usage metadata:", response.usage_metadata)


if __name__ == "__main__":
    asyncio.run(main())
