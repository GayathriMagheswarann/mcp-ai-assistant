from mcp.server import MCPServer

mcp = MCPServer("Basic Server")


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b