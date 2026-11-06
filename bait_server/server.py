"""MCP server that serves an attack's simulated tools and records every call."""

import asyncio
import json
import os
from pathlib import Path

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

TOOLS_CHANGED = "__tools_changed__"  # Logged when an agent sees a changed tool description.
GUARD = "__guard__"  # Logged for every text a guard checks.
BLOCKED_DESCRIPTION = "[Description removed by a security filter.]"
BLOCKED_OUTPUT = "[Tool output blocked by a security filter: possible prompt injection.]"


def tool_text(description: str, parameters: dict) -> str:
    """What a guard reads for a tool. Parameter descriptions count, since attacks can hide there."""
    return "\n".join([description, *(f"{name}: {text}" for name, text in parameters.items())])


class BaitTools:
    def __init__(self, attack: dict, log_path: Path, guard=None):
        self.tools = {tool["name"]: tool for tool in attack["tools"]}
        self.log_path = log_path
        self.call_counts: dict[str, int] = {}
        self.guard = guard
        self.payload = attack.get("payload")  # The attacker's text, so blocks of anything else count as false alarms.
        self._verdicts: dict[str, bool] = {}  # Agents re-list tools every step, so check each text once.

    def record(self, tool: str, args: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"tool": tool, "args": args}) + "\n")

    def blocked(self, text: str, where: str, tool: str) -> bool:
        """Ask the guard about `text`, logging the verdict the first time it's seen."""
        if self.guard is None:
            return False
        if text not in self._verdicts:
            verdict = self.guard.check(text)
            self._verdicts[text] = verdict.is_attack
            self.record(
                GUARD,
                {
                    "where": where,
                    "tool": tool,
                    "blocked": verdict.is_attack,
                    "payload": bool(self.payload) and self.payload in text,
                    "score": round(verdict.score, 4),
                    "ms": round(verdict.ms, 1),
                },
            )
        return self._verdicts[text]

    def _description(self, tool: dict) -> str:
        changed = tool.get("description_after_first_call")
        if changed and self.call_counts.get(tool["name"]):
            self.record(TOOLS_CHANGED, {"tool": tool["name"]})
            return changed
        return tool["description"]

    def _tool(self, tool: dict) -> types.Tool:
        description = self._description(tool)
        parameters = dict(tool.get("parameters", {}))
        if self.blocked(tool_text(description, parameters), "description", tool["name"]):
            description = BLOCKED_DESCRIPTION
            parameters = {name: "" for name in parameters}
        return types.Tool(
            name=tool["name"],
            description=description,
            input_schema={
                "type": "object",
                "properties": {
                    name: {"type": "string", "description": text} for name, text in parameters.items()
                },
            },
        )

    def list_tools(self) -> list[types.Tool]:
        return [self._tool(tool) for tool in self.tools.values()]

    def call_tool(self, name: str, args: dict) -> types.CallToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
                is_error=True,
            )
        self.record(name, args)
        count = self.call_counts.get(name, 0)
        self.call_counts[name] = count + 1
        output = tool["outputs"][min(count, len(tool["outputs"]) - 1)]
        if self.blocked(output, "output", name):
            output = BLOCKED_OUTPUT
        return types.CallToolResult(content=[types.TextContent(type="text", text=output)])


def build_server(bait: BaitTools) -> Server:
    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=bait.list_tools())

    async def on_call_tool(ctx, params) -> types.CallToolResult:
        return bait.call_tool(params.name, params.arguments or {})

    return Server("bait-server", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def main() -> None:
    attack = json.loads(Path(os.environ["BAIT_ATTACK_FILE"]).read_text(encoding="utf-8"))
    guard_spec = os.environ.get("BAIT_GUARD")
    guard = None
    if guard_spec:
        from guards import load_guard

        guard = load_guard(guard_spec)
    server = build_server(BaitTools(attack, Path(os.environ["BAIT_CALL_LOG"]), guard))
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
