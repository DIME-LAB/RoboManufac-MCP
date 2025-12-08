from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Server")

@mcp.tool()
def get_weather(location: str) -> str:
    """
    Gets the weather given a location
    Args:
        location: location, can be city, country, state, etc.
    """
    return f"The weather in {location} is hot and dry"

if __name__ == "__main__":
    mcp.run()