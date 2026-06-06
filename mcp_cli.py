#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp>=1.0",
#   "click",
#   "rich",
#   "pyyaml",
#   "platformdirs",
# ]
# ///
"""
mcp-cli — Generic CLI for MCP (Model Context Protocol) servers.

Configure servers in mcp-cli.yaml, then run:
  mcp-cli <server> <tool> [--option value ...]
"""

__version__ = "0.1.0"

import asyncio
import concurrent.futures
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import click
import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from platformdirs import user_config_dir
from mcp.shared.exceptions import McpError
from rich.console import Console
from rich.json import JSON

console = Console()

# ============================================================================
# Configuration
# ============================================================================

_CONFIG_SEARCH_PATHS = [
    ".mcp-cli.yaml",
    ".mcp-cli.yml",
    "mcp-cli.yaml",
    "mcp-cli.yml",
]

# Platform-appropriate config dir: ~/.config/mcp-cli (Linux),
# ~/Library/Application Support/mcp-cli (macOS),
# %APPDATA%\mcp-cli (Windows).
_USER_CONFIG_DIR = Path(user_config_dir("mcp-cli"))


def _find_config_file() -> Optional[Path]:
    if env_path := os.environ.get("MCP_CLI_CONFIG"):
        p = Path(env_path)
        if p.exists():
            return p
        click.echo(f"Warning: MCP_CLI_CONFIG={env_path} not found", err=True)

    for name in _CONFIG_SEARCH_PATHS:
        if (p := Path(name)).exists():
            return p

    for candidate in [
        _USER_CONFIG_DIR / "config.yaml",
        Path.home() / ".mcp-cli.yaml",
    ]:
        if candidate.exists():
            return candidate

    return None


def _load_yaml_servers() -> dict[str, dict]:
    """Return the servers dict from the config file, or {}."""
    config_file = _find_config_file()
    if not config_file:
        return {}
    with open(config_file) as f:
        data = yaml.safe_load(f) or {}
    return data.get("servers", {})


def list_servers() -> list[str]:
    """Return all server names from config file + env vars."""
    servers = list(_load_yaml_servers().keys())

    # Servers defined purely via env: MCP_CLI_<NAME>_URL=...
    for key in os.environ:
        if key.startswith("MCP_CLI_") and key.endswith("_URL"):
            name = key[len("MCP_CLI_"):-len("_URL")].lower()
            if name not in servers:
                servers.append(name)

    return servers


def load_server_config(server_name: str) -> dict:
    """Build config for a server from YAML + env var overrides."""
    config = dict(_load_yaml_servers().get(server_name, {}))

    # Env overrides: MCP_CLI_<SERVERNAME>_<KEY>=value
    prefix = f"MCP_CLI_{server_name.upper()}_"
    for key, val in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            config[config_key] = val

    return config


# ============================================================================
# MCP Client
# ============================================================================

@asynccontextmanager
async def _http_session(url: str, token: Optional[str]):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def _sse_session(url: str, token: Optional[str]):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with sse_client(url, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def _stdio_session(command: str, args: list[str], env: dict):
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def create_session(server_config: dict):
    """Create an MCP session based on transport type in config."""
    transport = server_config.get("transport", "http")

    if transport == "stdio":
        command = server_config.get("command")
        if not command:
            raise click.ClickException("stdio server config requires 'command'")
        args = server_config.get("args", [])
        env = {**os.environ, **server_config.get("env", {})}
        async with _stdio_session(command, args, env) as session:
            yield session
    elif transport in ("http", "sse"):
        url = server_config.get("url")
        if not url:
            raise click.ClickException(f"{transport.upper()} server config requires 'url'")
        token = None
        if token_env := server_config.get("token_env"):
            token = os.environ.get(token_env)
        if not token:
            token = server_config.get("token")
        session_fn = _sse_session if transport == "sse" else _http_session
        async with session_fn(url, token) as session:
            yield session
    else:
        raise click.ClickException(f"Unknown transport '{transport}'. Use http, sse, or stdio.")


async def list_tools(server_config: dict) -> list[dict[str, Any]]:
    async with create_session(server_config) as session:
        result = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            }
            for tool in result.tools
        ]


async def call_tool(server_config: dict, name: str, arguments: dict[str, Any]) -> Any:
    async with create_session(server_config) as session:
        return await session.call_tool(name, arguments)


# ============================================================================
# CLI Helpers
# ============================================================================

def run_async(coro):
    """Run a coroutine, handling the case where a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, coro).result()


def print_result(result, use_rich: bool = False):
    if hasattr(result, "content"):
        for item in result.content:
            if hasattr(item, "text"):
                if use_rich:
                    try:
                        console.print(JSON(item.text))
                    except Exception:
                        console.print(item.text)
                else:
                    click.echo(item.text)
    else:
        text = json.dumps(result, indent=2)
        console.print(JSON(text)) if use_rich else click.echo(text)


_TYPE_MAP = {
    "integer": click.INT,
    "number": click.FLOAT,
    "boolean": click.BOOL,
}


def _find_mcp_errors(exc) -> list:
    if isinstance(exc, McpError):
        return [exc]
    if isinstance(exc, BaseExceptionGroup):
        errors = []
        for sub in exc.exceptions:
            errors.extend(_find_mcp_errors(sub))
        return errors
    return []


def _parse_arg(value, schema_type: Optional[str]):
    if schema_type in ("array", "object") and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def build_command(tool: dict, server_config: dict) -> click.Command:
    """Build a Click command from an MCP tool definition."""
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    # Click normalizes a long option like ``--foo-bar`` to the param key
    # ``foo_bar``. Schema property names may themselves contain hyphens, so keep
    # a map from the normalized key back to the original name to avoid sending
    # the tool a mangled argument name (and to look up the right type).
    key_to_name = {name.replace("-", "_"): name for name in props}

    params = [
        click.Option(
            [f"--{name.replace('_', '-')}"],
            type=_TYPE_MAP.get(p.get("type"), click.STRING),
            required=name in required,
            help=p.get("description", ""),
        )
        for name, p in props.items()
    ]

    def callback(**kwargs):
        use_rich = click.get_current_context().find_root().params.get("use_rich", False)
        args = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            name = key_to_name.get(k, k)
            args[name] = _parse_arg(v, props.get(name, {}).get("type"))
        try:
            result = run_async(call_tool(server_config, tool["name"], args))
        except BaseException as e:
            mcp_errors = _find_mcp_errors(e)
            if mcp_errors:
                for err in mcp_errors:
                    click.echo(f"Error: {err}", err=True)
                sys.exit(1)
            raise
        print_result(result, use_rich=use_rich)

    return click.Command(
        name=tool["name"],
        callback=callback,
        params=params,
        help=tool.get("description", ""),
    )


# ============================================================================
# Dynamic CLI Groups
# ============================================================================

class ServerGroup(click.Group):
    """Click group whose commands are loaded dynamically from one MCP server."""

    def __init__(self, server_name: str, server_config: dict, **kwargs):
        super().__init__(**kwargs)
        self._server_name = server_name
        self._server_config = server_config
        self._tools: Optional[dict] = None

    @property
    def tools(self) -> dict:
        if self._tools is None:
            try:
                self._tools = {
                    t["name"]: t
                    for t in run_async(list_tools(self._server_config))
                }
            except Exception as e:
                click.echo(f"Error connecting to '{self._server_name}': {e}", err=True)
                self._tools = {}
        return self._tools

    def list_commands(self, ctx) -> list[str]:
        return sorted(self.tools.keys())

    def get_command(self, ctx, name) -> Optional[click.Command]:
        if name not in self.tools:
            return None
        return build_command(self.tools[name], self._server_config)


class MCPMultiServerCLI(click.Group):
    """Top-level CLI that exposes each configured MCP server as a subcommand group."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._servers: Optional[list[str]] = None

    @property
    def servers(self) -> list[str]:
        if self._servers is None:
            self._servers = list_servers()
        return self._servers

    def list_commands(self, ctx) -> list[str]:
        servers = self.servers
        if not servers:
            click.echo(
                "No servers configured. Create a mcp-cli.yaml or set MCP_CLI_<NAME>_URL.",
                err=True,
            )
        return sorted(servers)

    def get_command(self, ctx, name) -> Optional[click.Group]:
        if name not in self.servers:
            return None
        config = load_server_config(name)
        if not config:
            click.echo(
                f"No configuration found for server '{name}'. Check your mcp-cli.yaml.",
                err=True,
            )
            return None
        return ServerGroup(
            name=name,
            server_name=name,
            server_config=config,
            help=f"Tools from the '{name}' MCP server.",
        )


# ============================================================================
# Entry Point
# ============================================================================

@click.group(cls=MCPMultiServerCLI)
@click.version_option(version=__version__)
@click.option("--rich", "use_rich", is_flag=True, default=False, help="Enable rich terminal output.")
def main(use_rich):
    """mcp-cli — Generic CLI for MCP (Model Context Protocol) servers.

    Each configured server is a subcommand group; tools within it are sub-subcommands.
    Parameters are generated dynamically from each server's tool schemas.

    \b
    Usage:
      mcp-cli <server> <tool> [--option value ...]

    \b
    Config file (searched in order):
      $MCP_CLI_CONFIG              env var pointing to a YAML file
      .mcp-cli.yaml                current directory
      <platform config dir>/mcp-cli/config.yaml
      ~/.mcp-cli.yaml

    \b
    Or configure via environment variables alone:
      MCP_CLI_<NAME>_URL=https://...
      MCP_CLI_<NAME>_TOKEN=secret

    \b
    Supported transports (default: http):
      http    Streamable HTTP (MCP spec 2025-03-26+)
      sse     Server-Sent Events (older MCP servers)
      stdio   Subprocess over stdin/stdout
    """


if __name__ == "__main__":
    main()
