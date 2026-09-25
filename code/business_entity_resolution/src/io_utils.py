"""Streaming TSV IO. Every file in this contest is tab-separated; addresses and
ID lists contain commas, so a plain read collapses each row into one column."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(r"D:\ML challenge\dataset\student_resource\dataset")


def iter_source(path: Path):
    """Yield (entity_id, business_name, business_address, country)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            while len(parts) < 4:
                parts.append("")
            yield parts[0], parts[1], parts[2], parts[3]


def iter_ground_truth(path: Path):
    """Yield (source1_entity_id, set_of_matched_ids)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            raw = parts[1] if len(parts) > 1 else ""
            ids = {x for x in raw.split(",") if x.strip()} if raw.strip() else set()
            yield sid, ids


def write_submission(path: Path, header_second: str, rows: dict[str, list[str]], order: list[str]):
    """Write one row per Source-1 id, in `order`, tab separated, comma joined ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_second}\n")
        for sid in order:
            ids = rows.get(sid, [])
            seen, uniq = set(), []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            f.write(sid + "\t" + ",".join(uniq) + "\n")
