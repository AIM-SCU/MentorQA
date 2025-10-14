import json
from pathlib import Path

def pick20_min_overlap(json_path="finalQA.json"):
    """
    Read finalQA.json, keep all items if <=20,
    otherwise pick exactly 20 QAs minimizing overlap.
    Overwrites the same file (creates .bak backup).
    """

    START_KEYS = ["starting_sentence", "start_sentence", "start", "s"]
    END_KEYS   = ["ending_sentence",   "end_sentence",   "end",   "e"]
    SEG_KEYS   = ["segment_id", "group_id", "segment", "group", "g"]
    ID_KEYS    = ["id", "qid", "qa_id"]

    def get_first(d, keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    def extract(item, idx):
        s = int(get_first(item, START_KEYS))
        e = int(get_first(item, END_KEYS))
        if e < s: s, e = e, s
        seg = get_first(item, SEG_KEYS, None)
        tid = get_first(item, ID_KEYS, idx)
        return (s, e, seg, tid)

    def overlap(a, b):
        s1, e1 = a; s2, e2 = b
        return max(0, min(e1, e2) - max(s1, s2) + 1)

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
            "len": e - s + 1,
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
            key = (ov, -delta, Di, Li, f["tid"])
            if best is None or key < best[0]:
                best = (key, f)
        chosen = best[1]
        selected.append(chosen["idx"])
        sel_intervals.append(chosen["interval"])
        sel_segs.append(chosen["seg"])

    selected.sort()
    new_items = [items[i] for i in selected]

    print(f"✅ Picked {len(new_items)} minimally overlapping QAs from {n} total.")
    return new_items

    #backup = path.with_suffix(path.suffix + ".bak")
    #backup.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # if isinstance(data, list):
    #     out = new_items
    # else:
    #     out = dict(data)
    #     for k, v in out.items():
    #         if isinstance(v, list):
    #             out[k] = new_items
    #             break

    # path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    # print(f"Wrote {len(new_items)} items to {path} (backup saved as {backup})")