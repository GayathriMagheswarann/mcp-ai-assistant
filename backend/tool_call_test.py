from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage

from mcp_tools import search_files, read_file


llm = ChatOllama(
    model="qwen3:4b"
)

tools = [
    search_files,
    read_file
]

llm_with_tools = llm.bind_tools(tools)


messages = [
    (
        "system",
        "You are an AI assistant that can access study notes "
        "using the available tools. "
        "Use the tools when necessary, and after obtaining "
        "the required information, give the user a clear answer."
    ),
    (
        "human",
        "Find my Unit 1 notes."
    ),
]


while True:

    response = llm_with_tools.invoke(messages)

    print("\nModel response:")
    print(response)

    # Add the model's response to the conversation
    messages.append(response)

    # If the model does not request any tools,
    # we have reached the final answer.
    if not response.tool_calls:
        print("\nFinal AI response:")
        print(response.content)
        break

    # Execute every tool requested by the model
    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print("\nTool requested:")
        print(tool_name)
        print(tool_args)

        if tool_name == "search_files":
            result = search_files.invoke(tool_args)

        elif tool_name == "read_file":
            result = read_file.invoke(tool_args)

        else:
            result = "Unknown tool."

        print("\nMCP tool result:")
        print(result)

        # Give the tool result back to the model
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )