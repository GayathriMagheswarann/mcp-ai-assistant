
import asyncio

from langchain_core.messages import HumanMessage

from langgraph_agent import graph


async def main():
    print("Testing LangGraph MCP tool calling...")

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Call search_file_content with query='Python' and filename='PYTHON INTRO.txt'. Then summarize the returned passages."
                    )
                )
            ]
        }
    )

    print("\n--- Final conversation ---")

    for message in result["messages"]:
        print(f"\nMessage type: {type(message).__name__}")

        if getattr(message, "tool_calls", None):
            print("Tool calls requested:", message.tool_calls)

        if getattr(message, "name", None):
            print("Tool name:", message.name)

        if getattr(message, "content", None):
            print("Content:", message.content)


if __name__ == "__main__":
    asyncio.run(main())
