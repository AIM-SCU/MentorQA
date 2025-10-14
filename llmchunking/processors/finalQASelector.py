import json
from pathlib import Path

def pick20_min_overlap(json_path="finalQA.json"):
    """
    Read finalQA.json, keep all items if <=20,
    otherwise pick exactly 20 QAs minimizing time overlap.
    Overwrites the same file (creates .bak backup).
    """

    START_KEYS = ["starting_line", "start_line", "start", "s", "time_start_sec"]
    END_KEYS   = ["ending_line", "end_line", "end", "e", "time_end_sec"]
    SEG_KEYS   = ["segment_id", "group_id", "segment", "group", "g", "topic"]
    ID_KEYS    = ["id", "qid", "qa_id"]

    def get_first(d, keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    def extract(item, idx):
        s = get_first(item, START_KEYS)
        e = get_first(item, END_KEYS)

        # fallback if start/end missing
        try:
            s = float(s)
        except (TypeError, ValueError):
            s = float(idx)
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = s + 1.0  # assume minimal nonzero duration

        if e < s:
            s, e = e, s

        seg = get_first(item, SEG_KEYS, None)
        tid = get_first(item, ID_KEYS, idx)
        return (s, e, seg, tid)

    def overlap(a, b):
        s1, e1 = a; s2, e2 = b
        return max(0, min(e1, e2) - max(s1, s2))

    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    items = data if isinstance(data, list) else next(
        v for v in data.values() if isinstance(v, list)
    )

    n = len(items)
    if n <= 20:
        print(f"{n} items ≤ 20 — keeping all.")
        return items

    fields = []
    for i, it in enumerate(items):
        s, e, seg, tid = extract(it, i)
        fields.append({
            "idx": i,
            "interval": (s, e),
            "seg": seg,
            "len": e - s,
            "tid": tid
        })

    selected, sel_intervals, sel_segs = [], [], []

    while len(selected) < 20:
        best = None
        for f in fields:
            if f["idx"] in selected: continue
            ov = sum(overlap(f["interval"], iv) for iv in sel_intervals)
            Li = f["len"]
            delta = Li - ov
            Di = sum(1 for sg in sel_segs if sg == f["seg"])
            key = (ov, -delta, Di, -Li, f["tid"])
            if best is None or key < best[0]:
                best = (key, f)
        if best is None:
            break
        chosen = best[1]
        selected.append(chosen["idx"])
        sel_intervals.append(chosen["interval"])
        sel_segs.append(chosen["seg"])

    selected.sort()
    new_items = [items[i] for i in selected]

    # Backup
    # backup = path.with_suffix(path.suffix + ".bak")
    # backup.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Overwrite
    #path.write_text(json.dumps(new_items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Picked {len(new_items)} minimally overlapping QAs from {n} total.")
    #print(f"💾 Overwritten {json_path} (backup at {backup})")

    return new_items