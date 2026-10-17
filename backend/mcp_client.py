import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="uv",
    args=[
        "run",
        "--with",
        "mcp==2.2.0",
        "mcp",
        "run",
        "D:/PROJECTS/mcp-ai-assistant/backend/file_server.py",
    ],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            # Discover available tools
            tools = await session.list_tools()

            print("Available tools:")

            for tool in tools.tools:
                print(f"- {tool.name}")

            # Call the search_files tool
            result = await session.call_tool(
                "search_files",
                arguments={"query": "Unit"},
            )

            print("\nSearch result:")
            print(result)

            # Call the read_file tool
            file_result = await session.call_tool(
                "read_file",
                arguments={"filename": "MPC-UNIT 1.pdf"},
            )

            print("\nFile content:")
            print(file_result)


if __name__ == "__main__":
    asyncio.run(main())