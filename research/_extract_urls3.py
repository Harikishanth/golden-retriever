import re
from pathlib import Path

s = Path(r"d:\ML challenge\research\_builder_index.js").read_text(encoding="utf-8", errors="ignore")
idx = s.find("api.cosmic.aws.dev")
print("idx", idx)
print(s[idx-200:idx+400])
print("==== more cosmic ====")
for m in re.finditer(r".{80}cosmic.{80}", s):
    print(m.group()[:200])
    print("---")
