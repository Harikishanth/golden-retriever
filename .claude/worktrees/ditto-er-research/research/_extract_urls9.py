from pathlib import Path
s = Path(r"d:\ML challenge\research\_client.js").read_text(encoding="utf-8", errors="ignore")
# find GetArticleV2 http definition
for needle in ["GetArticleV2", "/cs/v2/articles", "GetContent`"]:
    i = 0
    c = 0
    while c < 3:
        i = s.find(needle, i)
        if i < 0:
            break
        print("====", needle, i)
        print(s[max(0,i-120):i+250])
        print()
        i += len(needle)
        c += 1
