import re
from pathlib import Path
s = Path(r"d:\ML challenge\research\_builder_index.js").read_text(encoding="utf-8", errors="ignore")
ops = set(re.findall(r"`([A-Z][A-Za-z0-9]{6,70})`", s))
for o in sorted(ops):
    lo = o.lower()
    if any(x in lo for x in ["content", "space", "mdx", "publish", "article", "slug", "search"]):
        print(o)
print("COUNT", len(ops))
# http paths in smithy: POST`,`/foo
http = set(re.findall(r"\[`(GET|POST|PUT|DELETE)`,`([^`]+)`", s))
print("HTTP", sorted(http)[:80])
