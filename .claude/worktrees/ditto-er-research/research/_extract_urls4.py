import re
from pathlib import Path

s = Path(r"d:\ML challenge\research\_builder_index.js").read_text(encoding="utf-8", errors="ignore")

# smithy-style operation names often appear as PascalCase strings
ops = set(re.findall(r"[\"']([A-Z][A-Za-z]{8,60})[\"']", s))
for o in sorted(ops):
    if any(x in o.lower() for x in ["content", "article", "post", "space", "mdx", "publish"]):
        print(o)

print("==== HTTP paths ====")
paths = set(re.findall(r"[\"'](/[a-zA-Z0-9_{}/-]{5,80})[\"']", s))
for p in sorted(paths):
    print(p)
