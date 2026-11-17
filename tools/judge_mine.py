"""ML-4 judge calibration: the weights (W_COLOR, W_EDGE, W_PIECE) and JUDGE_MARGIN mined on the photos.

Spec ruling 3: the judge is calibrated so that candidate 0 (the rules head) wins on the known-good
heads: the 24 photos outside the 52 "not working" list plus the 13 of the 52 judged a match by eye
(37 heads). The references are not used for that count; among the other 39 heads the change of the
layer IoU against the reference (tools/score.py's shape_compare) when an alternative wins is reported
and only breaks ties (the references are AI-made: never optimised for alone).

Phases (each cached under --data, default .cache/judge_mine; reader cache warm, ROSETTE_READER=cached):
  collect  per photo, analyze_with_fit as tools/score.py runs it (new image: own design left out),
           with the judge's choice forced to candidate 0; records every candidate's three terms
           (color, edge, piece), its changed parts and label, and candidate 0's layer IoU
  force    for each of the 39 and each candidate that wins under some grid setting, the same run
           with that candidate forced: its layer IoU
  grid     every (W_COLOR, W_EDGE, W_PIECE) in steps of 0.1 summing to 1 times JUDGE_MARGIN in
           MARGINS, scored from the recorded terms (judge.combine, judge.choose: the judge's own
           arithmetic); prints the table, the leave-one-out count and the chosen setting

Usage: .venv/Scripts/python.exe tools/judge_mine.py [collect|force|grid ...] [--data DIR] [--ids ID ...]
(no phase: all three in order; a phase reuses the cached rows of earlier runs, --ids limits collect/force)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("ROSETTE_READER", "cached")  # never starts the model

from rosette.vision import judge as vjudge  # noqa: E402

PHOTOS = ROOT / "reference" / "photos"
REF_SVG = ROOT / "reference" / "svg"
# The 52 heads the 2026-11 review called not working (ledger, outline52 pages).
NOT_WORKING = (
    "A10-15B A12-15 A14-17 A15-17 A16-17 A3-15 S10-19 S11-15 S1-15 S1-15B S11a-15 S13-15 S15-15 S15-15B "
    "S17-15 S18-15 S18-15B S2-15 S2-15B S26-15 S27-15 S28-15 S31-15 S3-15 S32-15 S33-15 S36-15B S37-15 "
    "S38-15 s39-15 S40-15 S41-15 S4-15 S42-15 S43-15 s44-15 S45-15 S46-15 S47-15 S48-17 S49-17 S5-15 "
    "S53-19 S55-15 S58-17 S6-15 S63-17 S65-18 S67-19 S68-19 S69-19 S70-19").split()
# Of the 52, judged a match by eye on the head-zoom page (ledger).
MATCH_BY_EYE = ("S18-15 S18-15B S1-15B S31-15 S36-15B S41-15 S53-19 S55-15 S6-15 S65-18 S69-19 S70-19 "
                "s39-15").split()
MARGINS = (0.005, 0.01, 0.02)
STEPS = 10  # weights in steps of 1/STEPS
IOU_EPS = 0.001  # a layer IoU change smaller than this is "same"


def all_ids() -> list[str]:
    return sorted(f.stem for f in PHOTOS.glob("*.png"))


def known_good() -> list[str]:
    bad = set(NOT_WORKING)
    return sorted([i for i in all_ids() if i not in bad] + MATCH_BY_EYE)


def others() -> list[str]:
    good = set(known_good())
    return [i for i in all_ids() if i not in good]


def grid() -> list[tuple[tuple[float, float, float], float]]:
    out = []
    for a in range(STEPS + 1):
        for b in range(STEPS + 1 - a):
            w = (a / STEPS, b / STEPS, (STEPS - a - b) / STEPS)
            out.extend((w, m) for m in MARGINS)
    return out


def decide(row: dict, weights, margin: float) -> int:
    """The judge's winner for one recorded head under (weights, margin); 0 when the judge did not run."""
    cands = row.get("candidates") or []
    if len(cands) < 2 or row.get("skipped"):
        return 0
    scores = [vjudge.combine(c["terms"], weights) for c in cands]
    return vjudge.choose(scores, lambda i: cands[i]["changed"], margin)[0]


# ---- runs (collect, force) -------------------------------------------------------------------------

class _Recorder:
    """Wraps vjudge.judge: records every call's candidates and forces the winner (index or None)."""

    def __init__(self):
        self.calls: list[dict] = []
        self.force: int | None = None
        self.orig = vjudge.judge

    def __call__(self, candidates, p, labels, pieces, decisions=None, merged=None):
        v = self.orig(candidates, p, labels, pieces, decisions=decisions, merged=merged)
        self.calls.append(dict(
            labels=list(v.labels), skipped=bool(v.note),
            candidates=[dict(label=lb, terms=dict(t), score=float(s),
                             changed=vjudge.changed_parts(candidates[0], c) if i else [])
                        for i, (lb, t, s, c) in enumerate(zip(v.labels, v.terms, v.scores, candidates))]))
        i = self.force
        if i is None or i == v.winner:
            return v
        if i >= len(candidates):
            raise IndexError(f"forced candidate {i} of {len(candidates)}")
        if i == 0:
            return vjudge.Verdict(winner=0, scores=v.scores, terms=v.terms, margin=0.0, labels=v.labels)
        ids = {q.id for q in candidates[i].parts}
        return vjudge.Verdict(winner=i, scores=v.scores, terms=v.terms,
                              margin=max(float(v.scores[i] - v.scores[0]), vjudge.JUDGE_MARGIN), labels=v.labels,
                              changed=vjudge.changed_parts(candidates[0], candidates[i]),
                              removed=[q.id for q in candidates[0].parts if q.id not in ids])


def _run(sid: str, rec: _Recorder, force: int) -> dict:
    """analyze_with_fit (new image) with candidate `force` chosen; the judge's record and layer IoU."""
    from rosette.pipeline import analyze_with_fit, render
    from rosette.vision.templates import design_key
    from tools.score import shape_compare
    rec.calls, rec.force = [], force
    t = time.perf_counter()
    model, _ = analyze_with_fit((PHOTOS / f"{sid}.png").read_bytes(), exclude_key=design_key(sid))
    svg = render(model).svg
    seconds = time.perf_counter() - t
    ref = REF_SVG / f"{sid}.svg"
    layer = shape_compare(ref.read_text(encoding="utf-8"), svg)["layer_iou"] if ref.exists() else None
    if len(rec.calls) > 1:
        raise RuntimeError(f"{sid}: the judge ran {len(rec.calls)} times")
    call = rec.calls[0] if rec.calls else dict(labels=[], skipped=False, candidates=[])
    return dict(call, layer_iou=layer, seconds=round(seconds, 1))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def collect(data: Path, ids: list[str]) -> None:
    path = data / "terms.json"
    rows = _load(path)
    rec = _Recorder()
    vjudge.judge = rec
    try:
        for sid in ids:
            if sid in rows:
                continue
            row = _run(sid, rec, 0)
            rows[sid] = row
            _save(path, rows)
            print(f"collect {sid} {row['seconds']}s candidates {len(row['candidates'])} layer {row['layer_iou']}",
                  flush=True)
    finally:
        vjudge.judge = rec.orig


def force(data: Path, ids: list[str]) -> None:
    rows = _load(data / "terms.json")
    path = data / "ious.json"
    ious = _load(path)
    rec = _Recorder()
    vjudge.judge = rec
    try:
        for sid in ids:
            row = rows.get(sid)
            if row is None:
                print(f"force {sid}: not collected", flush=True)
                continue
            got = ious.setdefault(sid, {"0": row["layer_iou"]})
            wins = sorted({decide(row, w, m) for w, m in grid()} - {0})
            for i in wins:
                if str(i) in got:
                    continue
                out = _run(sid, rec, i)
                if out["labels"] != row["labels"]:
                    raise RuntimeError(f"{sid}: the candidates changed between runs")
                got[str(i)] = out["layer_iou"]
                _save(path, ious)
                print(f"force {sid} candidate {i} ({row['candidates'][i]['label']}) {out['seconds']}s "
                      f"layer {row['layer_iou']} -> {out['layer_iou']}", flush=True)
            _save(path, ious)
    finally:
        vjudge.judge = rec.orig


# ---- the grid --------------------------------------------------------------------------------------

def evaluate(rows: dict, ious: dict, weights, margin: float, good: list[str], rest: list[str]) -> dict:
    wins = [sid for sid in good if sid in rows and decide(rows[sid], weights, margin) == 0]
    rises = drops = same = changed = 0
    for sid in rest:
        if sid not in rows:
            continue
        w = decide(rows[sid], weights, margin)
        if w == 0:
            continue
        changed += 1
        got = ious.get(sid, {})
        a, b = got.get("0"), got.get(str(w))
        if a is None or b is None:
            continue
        if b > a + IOU_EPS:
            rises += 1
        elif b < a - IOU_EPS:
            drops += 1
        else:
            same += 1
    return dict(a=len(wins), wins=wins, rises=rises, drops=drops, same=same, changed=changed)


def _key(ev: dict, setting, a: int) -> tuple:
    (wc, we, wp), m = setting
    dist = abs(wc - vjudge.W_COLOR) + abs(we - vjudge.W_EDGE) + abs(wp - vjudge.W_PIECE) + abs(m - vjudge.JUDGE_MARGIN)
    return (-a, -ev["rises"], ev["drops"], round(dist, 6))


def mine(rows: dict, ious: dict) -> dict:
    good = [s for s in known_good() if s in rows]
    rest = [s for s in others() if s in rows]
    settings = grid()
    evs = [evaluate(rows, ious, w, m, good, rest) for w, m in settings]
    order = sorted(range(len(settings)), key=lambda k: _key(evs[k], settings[k], evs[k]["a"]))
    best = order[0]
    loo = []
    for h in good:  # the best setting on the other heads, applied to h
        def a_without(k):
            return evs[k]["a"] - (h in evs[k]["wins"])
        k = min(range(len(settings)), key=lambda k: _key(evs[k], settings[k], a_without(k)))
        loo.append((h, settings[k], h in evs[k]["wins"]))
    return dict(settings=settings, evs=evs, order=order, best=best, loo=loo, good=good, rest=rest)


def _fmt(setting) -> str:
    (wc, we, wp), m = setting
    return f"{wc:.1f} {we:.1f} {wp:.1f} {m:.3f}"


def report(rows: dict, ious: dict, top: int = 20) -> dict:
    r = mine(rows, ious)
    settings, evs = r["settings"], r["evs"]
    cur = next(k for k, s in enumerate(settings)
               if s == ((vjudge.W_COLOR, vjudge.W_EDGE, vjudge.W_PIECE), vjudge.JUDGE_MARGIN))
    print(f"heads: {len(r['good'])} known-good, {len(r['rest'])} others; grid {len(settings)} settings")
    trivial = sum(len(rows[s].get("candidates") or []) < 2 or rows[s].get("skipped") for s in r["good"])
    print(f"known-good heads the judge cannot change (no open decision, one candidate or skipped): {trivial}")
    print("| rank | W_COLOR W_EDGE W_PIECE MARGIN | (a) wins of 37 | others changed | IoU up | IoU down | same |")
    print("|---|---|---|---|---|---|---|")
    for n, k in enumerate(r["order"][:top], 1):
        e = evs[k]
        print(f"| {n} | {_fmt(settings[k])} | {e['a']} | {e['changed']} | {e['rises']} | {e['drops']} | {e['same']} |")
    e = evs[cur]
    print(f"| current | {_fmt(settings[cur])} | {e['a']} | {e['changed']} | {e['rises']} | {e['drops']} | {e['same']} |")
    loo_wins = sum(ok for _, _, ok in r["loo"])
    print(f"leave-one-out: {loo_wins}/{len(r['loo'])} known-good heads keep candidate 0 under the best "
          "setting of the other heads")
    for h, s, ok in r["loo"]:
        if not ok or s != settings[r["best"]]:
            print(f"  {h}: setting {_fmt(s)} -> {'candidate 0' if ok else 'an alternative'}")
    best = settings[r["best"]]
    print(f"chosen: W_COLOR, W_EDGE, W_PIECE = {best[0][0]:.1f}, {best[0][1]:.1f}, {best[0][2]:.1f}; "
          f"JUDGE_MARGIN = {best[1]}")
    lost = [s for s in r["good"] if s not in evs[r["best"]]["wins"]]
    print(f"known-good heads an alternative wins under the chosen setting: {', '.join(lost) or 'none'}")
    print("per head under the chosen setting (non-trivial heads):")
    print("| ID | set | candidates | winner | label | margin | layer IoU 0 -> winner |")
    print("|---|---|---|---|---|---|---|")
    good = set(r["good"])
    for sid in sorted(rows):
        row = rows[sid]
        cands = row.get("candidates") or []
        if len(cands) < 2:
            continue
        w = decide(row, *best)
        scores = [vjudge.combine(c["terms"], best[0]) for c in cands]
        mg = vjudge.choose(scores, lambda i: cands[i]["changed"], best[1])[1]
        got = ious.get(sid, {})
        iou0, iouw = got.get("0", row.get("layer_iou")), got.get(str(w))
        f = (lambda v: "-" if v is None else f"{v:.3f}")
        print(f"| {sid} | {'good' if sid in good else 'other'}{' skipped' if row.get('skipped') else ''} | "
              f"{len(cands)} | {w} | {cands[w]['label']} | {mg:.4f} | {f(iou0)} -> {f(iouw if w else iou0)} |")
    return r


def main(argv: list[str]) -> int:
    data = ROOT / ".cache" / "judge_mine"
    if "--data" in argv:
        i = argv.index("--data")
        data = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    ids = None
    if "--ids" in argv:
        i = argv.index("--ids")
        ids, argv = argv[i + 1:], argv[:i]
    phases = argv or ["collect", "force", "grid"]
    for ph in phases:
        if ph == "collect":
            collect(data, ids or all_ids())
        elif ph == "force":
            force(data, ids or others())
        elif ph == "grid":
            report(_load(data / "terms.json"), _load(data / "ious.json"))
        else:
            print(f"unknown phase {ph}")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
