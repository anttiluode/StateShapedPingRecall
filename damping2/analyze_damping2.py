"""Summarise damping 2 against PROTOCOL_DAMPING2.md. Written before the batch results existed."""
import json, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
R = json.loads((HERE / "results_damping2.json").read_text()); rows = R["results"]
cell = lambda sp, inp: [x for x in rows if x["spacing"] == sp and x["input"] == inp]
med = lambda xs: float(np.median(xs))


def paired(a_rows, b_rows, fa, fb):
    """Per-seed mean of fa(a) minus fb(b); seed bootstrap as in Astra's analyze.py (rng 890, 20000)."""
    seeds = sorted({x["seed"] for x in a_rows})
    d = np.array([np.mean([fa(x) for x in a_rows if x["seed"] == s]) - np.mean([fb(x) for x in b_rows if x["seed"] == s])
                  for s in seeds])
    boot = d[np.random.default_rng(890).integers(0, len(d), (20000, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return {"mean_difference": float(d.mean()), "paired_seed_bootstrap_95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def block(rs):
    out = {"bodies": len(rs)}
    for v in ("trained", "destinations_shuffled", "timers_shuffled", "untrained"):
        out[v] = {k: med([x[v][k] for x in rs]) for k in
                  ("surprise_left", "surprise_left_first_event_window", "surprise_left_second_event_window",
                   "anti_image_score", "astra_reconstruction_score", "surprise_left_unexpected")}
        out[v]["own_preferred"] = int(sum(x[v]["own_preferred"] for x in rs))
    out["contrasts_control_minus_trained_surprise_left"] = {
        c: paired(rs, rs, lambda x, c=c: x[c]["surprise_left"], lambda x: x["trained"]["surprise_left"])
        for c in ("destinations_shuffled", "timers_shuffled", "untrained")}
    ex = [x for x in rs if x["bounds"].get("exact")]
    out["bounds_exact_bodies"] = len(ex)
    out["median_timing_blind_8"] = med([x["bounds"]["timing_blind_8"] for x in ex]) if ex else None
    out["median_ceiling_192"] = med([x["bounds"]["ceiling_192"] for x in ex]) if ex else None
    out["median_ceiling_weight_max_abs"] = med([x["bounds"]["ceiling_weight_max_abs"] for x in ex]) if ex else None
    out["rule_at_or_above_timing_blind"] = int(sum(x["bounds"]["local_rule_linear"] >= x["bounds"]["timing_blind_8"] for x in ex))
    out["median_timers_moved_by_own_weights"] = med([x["timers_moved_by_own_weights"] for x in rs])
    out["clipped_frames"] = int(sum(x["training_clipped_frames"] + x["trained"]["probe_clipped"] for x in rs))
    t, c = out["trained"], out["contrasts_control_minus_trained_surprise_left"]
    out["damping1_rule"] = {"1_surprise_left_at_most_0.75": t["surprise_left"] <= 0.75,
                            "2_anti_image_at_least_0.10": t["anti_image_score"] >= 0.10,
                            "3_own_preferred_at_least_18": t["own_preferred"] >= 18,
                            "4a_beats_destinations_shuffled": c["destinations_shuffled"]["paired_seed_bootstrap_95"][0] > 0,
                            "4b_beats_timers_shuffled": c["timers_shuffled"]["paired_seed_bootstrap_95"][0] > 0}
    out["damping1_rule"]["all_pass"] = all(out["damping1_rule"].values())
    return out


S = {"%s_%s" % c: block(cell(*c)) for c in R["cells"]}
ew, bs = cell("even", "world"), cell("bunched", "shared")
second = paired(bs, ew, lambda x: x["trained"]["surprise_left_second_event_window"],
                lambda x: x["trained"]["surprise_left_second_event_window"])
S["second_event_window_bunched_shared_minus_even_world"] = second
S["decision_rule"] = {"1_even_world_passes_all_four_damping1_conditions": S["even_world"]["damping1_rule"]["all_pass"],
                      "2_even_world_second_event_lower_than_bunched_shared": second["mean_difference"] > 0 and second["excludes_zero"],
                      "3_even_world_rule_at_or_above_timing_blind_in_18_of_24": S["even_world"]["rule_at_or_above_timing_blind"] >= 18}
S["decision_rule"]["diagnosis_confirmed"] = all(S["decision_rule"].values())
(HERE / "summary_damping2.json").write_text(json.dumps(S, indent=1) + "\n")
for k, v in S.items():
    if "trained" in v:
        t = v["trained"]
        print("%-15s left %.2f  first %.2f  second %.2f  anti %.2f  astra %.2f  own %d  unexp %.2f | timer-shuffle %s | blind %.2f ceil %.2f (|w| %.1f) rule>=blind %d/%d | moved %.0f clip %d | all4 %s" % (
            k, t["surprise_left"], t["surprise_left_first_event_window"], t["surprise_left_second_event_window"],
            t["anti_image_score"], t["astra_reconstruction_score"], t["own_preferred"], t["surprise_left_unexpected"],
            [round(x, 3) for x in v["contrasts_control_minus_trained_surprise_left"]["timers_shuffled"]["paired_seed_bootstrap_95"]],
            v["median_timing_blind_8"], v["median_ceiling_192"], v["median_ceiling_weight_max_abs"],
            v["rule_at_or_above_timing_blind"], v["bounds_exact_bodies"], v["median_timers_moved_by_own_weights"],
            v["clipped_frames"], v["damping1_rule"]["all_pass"]))
print("second-event window, bunched/shared minus even/world:", second)
print(json.dumps(S["decision_rule"], indent=1))
