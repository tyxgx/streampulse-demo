"""Generate precomputed answers for the demo's example chips (run against a local server with a key)."""
import json
import sys
import urllib.request

EXAMPLES = ["How is India doing?", "Total streams for United States in 2024", "Top 5 countries by streams",
            "Top 5 artists in 2024", "How many artists are in the data?", "What is the weather in Delhi?"]
base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
out = {}
for q in EXAMPLES:
    req = urllib.request.Request(f"{base}/api/ask", json.dumps({"question": q}).encode(), {"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=120))
    assert d["route"] in ("sql", "semantic", "gate"), (q, d)
    out[q.lower()] = {k: d[k] for k in ("answer", "route", "sources")}
    print(q, "->", d["route"])
json.dump(out, open("precomputed.json", "w"), indent=1)
