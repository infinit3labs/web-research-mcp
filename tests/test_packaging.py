"""Packaging/release checks: version sync, changelog discipline, and a
protocol smoke test against whatever `web-research-mcp` console script is
currently on PATH (installed by `pip install [-e] .` in CI or locally).

These guard the exact class of bug fixed in INF-243 (pyproject.toml and
__init__.py drifting apart) and the release automation gaps tracked in
INF-238.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match, "pyproject.toml is missing a top-level version"
    return match.group(1)


class VersionSyncTests(unittest.TestCase):
    def test_init_version_matches_pyproject(self):
        from web_research import __version__

        self.assertEqual(
            __version__, _pyproject_version(),
            "src/web_research/__init__.py.__version__ has drifted from pyproject.toml",
        )

    def test_changelog_has_heading_for_current_version(self):
        version = _pyproject_version()
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
        self.assertIn(
            f"## [{version}]", changelog,
            f"CHANGELOG.md has no dated heading for version {version} — "
            "move the Unreleased entry over as part of the release",
        )


class InstalledConsoleScriptSmokeTest(unittest.IsolatedAsyncioTestCase):
    """Exercises the `web-research-mcp` entry point exactly as an MCP client
    would launch it — via PATH, over stdio, no source-tree assumptions.

    Skips (rather than fails) when the console script isn't on PATH, since a
    plain checkout without `pip install [-e] .` never installs it. CI always
    installs the package before running tests, so this runs for real there.
    """

    async def test_console_script_speaks_mcp_protocol(self):
        server_bin = shutil.which("web-research-mcp")
        if not server_bin:
            self.skipTest("web-research-mcp console script not on PATH — package not installed")

        proc = await asyncio.create_subprocess_exec(
            server_bin,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async def send(msg: dict) -> dict:
                proc.stdin.write((json.dumps(msg) + "\n").encode())
                await proc.stdin.drain()
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=15.0)
                self.assertTrue(line, "server closed stdout without responding")
                return json.loads(line)

            init_resp = await send({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "packaging-smoke-test", "version": "0"},
                },
            })
            self.assertIn("result", init_resp, f"initialize failed: {init_resp}")

            proc.stdin.write((json.dumps({
                "jsonrpc": "2.0", "method": "notifications/initialized", "params": {},
            }) + "\n").encode())
            await proc.stdin.drain()

            list_resp = await send({
                "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
            })
            self.assertIn("result", list_resp, f"tools/list failed: {list_resp}")
            tools = {t["name"] for t in list_resp["result"]["tools"]}
            self.assertEqual(
                len(tools), 10,
                f"expected 10 registered tools from the installed console script, got {sorted(tools)}",
            )
        finally:
            proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()


if __name__ == "__main__":
    unittest.main()
