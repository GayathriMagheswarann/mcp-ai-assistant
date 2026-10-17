
import asyncio
from langchain_core.messages import HumanMessage
from langgraph_agent import llm_with_tools

async def main():
    system_prompt = (
    "system",
    "/no_think\n"
    "You are a tool-using AI assistant. "
    "When the user asks you to search for files, you MUST call "
    "the search_files tool. "
    "When the user asks to read a file, you MUST call "
    "the read_file tool. "
    "When the user asks to find information inside notes, you MUST call "
    "the search_file_content tool. "
    "Never describe a tool call in text instead of making the actual call. "
    "After receiving tool results, answer using those results. "
    "Do not narrate your reasoning. "
    "Return only the final answer."
),


    response = await llm_with_tools.ainvoke([
        ("system", system_prompt),
        HumanMessage(
            content=(
                "Use your available tools to search for my Python "
                "study files. Then tell me which files were found "
                "based on the tool results."
            )
        ),
    ])

    print("\nTool calls:", response.tool_calls)
    print("Finish reason:", response.response_metadata.get("done_reason"))
    print("Response content:", str(response.content)[:1000])

if __name__ == "__main__":
    asyncio.run(main())
