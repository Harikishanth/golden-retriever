from pathlib import Path
s = Path(r"d:\ML challenge\research\_client.js").read_text(encoding="utf-8", errors="ignore")
# vh is GetArticleV2 input codec - find `vh=`
for needle in ["vh=[", "vh=", "articleId", "contentId"]:
    i = s.find(needle)
    print(needle, i)
print("==== around Oy GetArticle ====")
i = s.find("Oy=[9,Y,ji")
print(s[i-200:i+200])
# search httpQuery bindings near articles
i = s.find("GET`,`/cs/v2/articles`")
print("==== binding ====")
print(s[i-80:i+120])
# find codec vh definition - might be var vh=
import re
m = re.search(r"\bvh=\[", s)
if m:
    print("vh def", s[m.start():m.start()+500])
