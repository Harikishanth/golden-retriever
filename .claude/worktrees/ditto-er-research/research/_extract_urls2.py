import re
from pathlib import Path

s = Path(r"d:\ML challenge\research\_builder_index.js").read_text(encoding="utf-8", errors="ignore")

# find nearby context for cosmic / content APIs
for pat in [
    r".{0,80}cosmic.{0,80}",
    r".{0,60}/contents.{0,60}",
    r".{0,60}GetPublished.{0,60}",
    r".{0,80}serviceEndpoint.{0,80}",
    r".{0,60}contentId.{0,80}",
    r".{0,40}mdxSource.{0,40}",
]:
    print("====", pat)
    hits = re.findall(pat, s, flags=re.I)
    for h in hits[:8]:
        print(h[:200])
        print("---")
