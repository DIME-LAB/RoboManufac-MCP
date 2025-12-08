# Installation Steps

To install the `math_tool` MCP server, run the following command:

```json
{
  "mcpServers": {
    "math_tool": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/inbarajaldrin/mcp-server-example.git",
        "mcp-server"
      ]
    }
  }
}
```

This will fetch and set up the `mcp-server` from the specified GitHub repository.