from pathlib import Path
s = Path(r"d:\ML challenge\research\_builder_index.js").read_text(encoding="utf-8", errors="ignore")
needle = "3HiM6zDmFrF98fRzOUETnGFDoqz"
print("article id in bundle", s.find(needle))
# look at Get* operations by scanning for `Get
import re
gets = set(re.findall(r"`(Get[A-Za-z]{4,50})`", s))
print("Get ops", sorted(gets)[:80])
posts = set(re.findall(r"`(/(?:content|contents|spaces|posts|articles|v1|v2)[^`]{0,80})`", s))
print("paths", sorted(posts)[:80])
slash = set(re.findall(r"`(/[a-z][a-zA-Z0-9_/{}\-]{4,90})`", s))
interesting = [p for p in sorted(slash) if any(x in p.lower() for x in ["content", "space", "post", "mdx", "publish", "article", "cosmic"])]
print("interesting", interesting[:100])
print("slash sample", list(sorted(slash))[:40])
