import re
from pathlib import Path
s = Path(r"d:\ML challenge\research\_builder_index.js").read_text(encoding="utf-8", errors="ignore")
files = re.findall(r"assets/[A-Za-z0-9_~.-]+\.js", s[:50000])
for f in sorted(set(files)):
    if any(x.lower() in f.lower() for x in ["content", "cosmic", "cms", "article", "mdx", "search"]):
        print(f)
print("total unique first chunk", len(set(files)))
# also scan whole file for ContentView or similar chunk names
files2 = set(re.findall(r"assets/[A-Za-z0-9_~.-]*[Cc]ontent[A-Za-z0-9_~.-]*\.js", s))
print("content files", sorted(files2))
