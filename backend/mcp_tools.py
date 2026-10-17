from langchain_core.tools import tool
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
async def call_mcp_tool(tool_name: str, arguments: dict):
    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments=arguments,
            )

            return result.structured_content["result"]
@tool
async def search_files(query: str) -> str:
    """Search the study notes folder for files matching a query."""

    result = await call_mcp_tool(
        "search_files",
        {"query": query},
    )
    return str(result)
@tool
async def read_file(filename: str) -> str:
    """Read a study notes text or PDF file."""

    result = await call_mcp_tool(
        "read_file",
        {"filename": filename},
    )
    return str(result)
@tool
async def search_file_content(query: str, filename: str = "") -> str:
    """Search inside study notes for passages relevant to a topic."""

    result = await call_mcp_tool(
        "search_file_content",
        {"query": query, "filename": filename},
    )
    return str(result)