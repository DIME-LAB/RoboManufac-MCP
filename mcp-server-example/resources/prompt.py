from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Prompts")

@mcp.prompt()
def get_prompt(topic: str) -> str:
    """
    Returns a prompt that will do detailed analysis of the given text.
    Args:
        topic: the topic to do research on
    """
    return f"Please do detailed research on the topic {topic} and return the results in a structured format."


if __name__ == "__main__":
    mcp.run()