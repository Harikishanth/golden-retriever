import re
from pathlib import Path

s = Path(r"d:\ML challenge\research\_builder_index.js").read_text(encoding="utf-8", errors="ignore")
urls = set(re.findall(r"https://[a-zA-Z0-9._/-]{8,160}", s))
for u in sorted(urls):
    lu = u.lower()
    if any(x in lu for x in ["api", "cosmic", "content", "graphql", "builder"]):
        print(u)
print("---PATHS---")
paths = set(re.findall(r"[\"'](/[a-zA-Z0-9_./-]{6,100})[\"']", s))
for u in sorted(paths):
    if any(x in u.lower() for x in ["content", "article", "post", "blog", "graphql", "cosmic"]):
        print(u)
print("---KEYWORDS---")
for kw in ["getContentById", "contentBySlug", "fetchContent", "contents/", "articles?"]:
    print(kw, s.find(kw))
