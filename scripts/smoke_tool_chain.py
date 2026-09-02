"""MCP-level smoke: research -> synthesize_report -> audit_citations chain."""
import asyncio
import json
import sys

sys.path.insert(0, "src")

from web_research import server


async def main() -> None:
    raw = await server.research("How do mRNA vaccines work?", depth="quick")
    m = raw.split("\n---\n\n", 1)[1]
    payload = json.loads(m)
    print("research: citations =", len(payload["citations"]))
    synth = await server.synthesize_report(m)
    print("synthesize_report:", synth.splitlines()[0])
    draft = (
        "# Report\n\nmRNA vaccines use lipid nanoparticles to deliver "
        "genetic instructions [1].\nThis sentence has no citation at all.\n"
    )
    audit = await server.audit_citations_tool(draft, "1,2,3")
    lines = audit.splitlines()
    verdict_line = lines[0]
    uncited = [l.strip() for l in lines if "Substantive" in l][0]
    print("audit_citations:", verdict_line, "|", uncited)


asyncio.run(main())
