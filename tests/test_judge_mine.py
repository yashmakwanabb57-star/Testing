"""ML-4 Task 6: the judge calibration's grid and decision replay (no model, no photo run)."""
from tools import judge_mine as M


def test_the_heads_split_into_37_known_good_and_39_others():
    good, rest = M.known_good(), M.others()
    assert len(M.all_ids()) == 76 and len(good) == 37 and len(rest) == 39
    assert set(M.MATCH_BY_EYE) <= set(good) and not set(good) & set(rest)


def test_the_grid_sums_to_one_in_tenths_times_three_margins():
    g = M.grid()
    assert len(g) == 66 * 3
    assert all(abs(sum(w) - 1.0) < 1e-9 for w, _ in g)
    assert {m for _, m in g} == set(M.MARGINS)


def test_decide_replays_the_judge_under_other_weights():
    row = {"candidates": [
        {"label": "rules head", "terms": {"color": 0.80, "edge": 0.40, "piece": 0.50}, "changed": []},
        {"label": "count p1 8", "terms": {"color": 0.78, "edge": 0.60, "piece": 0.50}, "changed": ["p1"]}]}
    assert M.decide(row, (1.0, 0.0, 0.0), 0.01) == 0  # colour alone: the rules head
    assert M.decide(row, (0.5, 0.5, 0.0), 0.01) == 1  # edges count: the alternative leads by 0.09
    assert M.decide(row, (0.5, 0.5, 0.0), 0.10) == 0  # under a wider margin
    assert M.decide(dict(row, skipped=True), (0.5, 0.5, 0.0), 0.01) == 0
    assert M.decide({"candidates": []}, (0.5, 0.5, 0.0), 0.01) == 0


def test_evaluate_counts_wins_and_the_layer_iou_change():
    rows = {"G": {"candidates": [{"terms": {"color": 0.9}, "changed": []},
                                 {"terms": {"color": 0.5}, "changed": ["p"]}]},
            "O": {"candidates": [{"terms": {"color": 0.5}, "changed": []},
                                 {"terms": {"color": 0.9}, "changed": ["p"]}]}}
    ev = M.evaluate(rows, {"O": {"0": 0.30, "1": 0.40}}, (1.0, 0.0, 0.0), 0.01, ["G"], ["O"])
    assert (ev["a"], ev["changed"], ev["rises"], ev["drops"]) == (1, 1, 1, 0)
