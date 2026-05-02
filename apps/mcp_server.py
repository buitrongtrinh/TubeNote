"""MCP stdio server entrypoint. Run: python -m apps.mcp_server"""
from link2slide.mcp.server import mcp


if __name__ == "__main__":
    mcp.run()
