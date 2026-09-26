import re
from pathlib import Path
s = Path(r"d:\ML challenge\research\_client.js").read_text(encoding="utf-8", errors="ignore")
http = re.findall(r"\[`(GET|POST|PUT|DELETE|PATCH)`,`([^`]+)`", s)
print("HTTP count", len(http))
for m, p in sorted(set(http), key=lambda x: x[1]):
    print(m, p)
print("==== GetArticle ====")
i = s.find("GetArticleV2")
print(s[i:i+400] if i>=0 else "none")
