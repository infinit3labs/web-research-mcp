#!/usr/bin/env python3
"""End-to-end MCP protocol test for the web-research server.

Uses proper MCP client semantics: keep stdin open, send requests one at a time,
wait for each response, only close stdin after the last response arrives.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Resolve the server launcher relative to this test file so the same test
# works in local development, in CI on a fresh checkout, and in any other
# execution context where the project lives somewhere other than the original
# author's machine.
REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = str(REPO_ROOT / "bin" / "web-research-mcp")


async def run():
    print(f"=== launching: {SERVER_BIN} ===", flush=True)
    proc = await asyncio.create_subprocess_exec(
        SERVER_BIN,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    responses_by_id: dict[int, dict] = {}
    notifications: list[dict] = []
    response_event = asyncio.Event()

    async def reader():
        while True:
            line = await proc.stdout.readline()
            if not line:
                # EOF — server closed its stdout
                response_event.set()
                return
            txt = line.decode().strip()
            if not txt:
                continue
            try:
                msg = json.loads(txt)
            except json.JSONDecodeError:
                print(f"  [non-json] {txt[:200]}")
                continue
            if "method" in msg and "id" not in msg:
                notifications.append(msg)
            elif "id" in msg:
                responses_by_id[msg["id"]] = msg
            response_event.set()

    async def send(msg: dict, expect_id: int, timeout: float = 30.0):
        """Send a message and wait for its response (or timeout)."""
        response_event.clear()
        proc.stdin.write((json.dumps(msg) + "\n").encode())
        await proc.stdin.drain()
        deadline = asyncio.get_event_loop().time() + timeout
        while expect_id not in responses_by_id:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"timeout waiting for response id={expect_id}")
            # Sleep briefly and let reader populate responses
            try:
                await asyncio.wait_for(response_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError(f"timeout waiting for response id={expect_id}")
            response_event.clear()
        return responses_by_id[expect_id]

    reader_task = asyncio.create_task(reader())

    try:
        # ---- 1. initialize ----
        init_resp = await send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "test-client", "version": "0.1"}},
        }, expect_id=1, timeout=15)
        assert "result" in init_resp, f"initialize failed: {init_resp}"
        sv = init_resp["result"].get("serverInfo", {})
        print(f"[1] initialize ✅  server: {sv.get('name')} @ {sv.get('version', '?')}")

        # ---- 2. initialized notification ----
        proc.stdin.write((json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized", "params": {}
        }) + "\n").encode())
        await proc.stdin.drain()

        # ---- 3. tools/list ----
        list_resp = await send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
        }, expect_id=2)
        assert "result" in list_resp, f"tools/list failed: {list_resp}"
        tools = list_resp["result"]["tools"]
        print(f"[2] tools/list ✅  {len(tools)} tools registered")
        expected = {
            "search_web", "fetch_url", "search_wikipedia", "search_academic",
            "search_news", "search_stackexchange", "search_scholar_meta",
        }
        got = {t["name"] for t in tools}
        missing = expected - got
        assert not missing, f"missing tools: {missing}"

        # ---- 4. tool calls, one at a time ----
        calls = [
            (10, "search_wikipedia",      {"query": "Model Context Protocol", "max_results": 2}),
            (11, "search_news",           {"query": "MCP server", "max_results": 3}),
            (12, "search_academic",       {"query": "transformer attention mechanism", "max_results": 2}),
            (13, "search_stackexchange",  {"query": "python asyncio gather", "max_results": 2, "site": "stackoverflow"}),
            (14, "search_scholar_meta",   {"query": "diffusion models image generation", "max_results": 2}),
            (15, "fetch_url",             {"url": "https://modelcontextprotocol.io/introduction"}),
            (16, "search_web (no keys)",  {"query": "test", "max_results": 3}),
        ]

        print()
        failures: list[str] = []
        for cid, name, args in calls:
            resp = await send({
                "jsonrpc": "2.0", "id": cid, "method": "tools/call",
                "params": {"name": name.split(" ")[0], "arguments": args},  # strip diagnostic suffix
            }, expect_id=cid, timeout=60)

            if "error" in resp:
                failures.append(f"[{cid}] {name}: error {resp['error']}")
                print(f"[{cid}] {name} ❌  error: {resp['error']}")
                continue
            content = resp["result"]["content"]
            if not content or content[0].get("type") != "text":
                failures.append(f"[{cid}] {name}: bad content shape")
                print(f"[{cid}] {name} ❌  bad shape")
                continue
            text = content[0]["text"]

            # Per-tool validation
            if name == "search_web (no keys)":
                # Two valid outcomes depending on whether API keys are configured:
                #   1) Empty-state message (no keys) — server should tell user how to enable
                #   2) Real results (at least one key configured) — server aggregates sources
                if ("No web results" in text and "API key" in text) or (
                    "Source:** search_web" in text and ("http" in text)
                ):
                    print(f"[{cid}] {name} ✅  ({'keyless empty-state' if 'API key' in text else 'real results with keys configured'})")
                else:
                    failures.append(f"[{cid}] {name}: unexpected keyless output: {text[:200]}")
                continue

            if "No " in text[:30] and "results" in text[:30]:
                failures.append(f"[{cid}] {name}: empty body: {text[:200]}")
                print(f"[{cid}] {name} ❌  empty body")
                continue

            has_signal = ("http" in text) or ("URL" in text) or ("Results for" in text) or ("Source:" in text)
            if not has_signal:
                failures.append(f"[{cid}] {name}: missing signal: {text[:200]}")
                print(f"[{cid}] {name} ❌  missing signal")
                continue

            preview = next((ln.strip() for ln in text.splitlines()
                            if ln.strip() and not ln.startswith("#")), "")
            print(f"[{cid}] {name} ✅  ({len(text):,} chars) — {preview[:100]}")

        # Graceful shutdown: close stdin, wait for server to exit
        print("\n--- closing stdin ---")
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            proc.kill()
        reader_task.cancel()

        stderr_text = (await proc.stderr.read()).decode().strip()
        print("\n=== STDERR ===")
        print(stderr_text or "(empty)")

        print()
        if failures:
            print(f"❌ {len(failures)} failures:")
            for f in failures:
                print(f"  - {f}")
            sys.exit(1)
        else:
            print("🎉 ALL TOOLS PASSED END-TO-END MCP PROTOCOL TEST")

    except Exception as e:
        print(f"TEST RUNNER ERROR: {type(e).__name__}: {e}")
        try:
            proc.kill()
        except Exception:
            pass
        reader_task.cancel()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run())
