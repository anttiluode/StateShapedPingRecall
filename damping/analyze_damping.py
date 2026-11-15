"""Summarise the damping batch; bootstrap over the 12 paired schedules, as in Astra's analyze.py."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
R = json.loads((HERE / "results_damping.json").read_text()); rows = R["results"]
sel = lambda task, jitter, sign=1.0: [x for x in rows if x["task"] == task and x["jitter"] == jitter and x["sign"] == sign]
med = lambda xs: float(np.median(xs))

def block(rs, three):
    out = {"bodies": len(rs), "schedules": len({x["seed"] for x in rs})}
    for v in ("trained", "destinations_shuffled", "timers_shuffled", "untrained"):
        if v not in rs[0]:
            continue
        out[v] = {"median_surprise_left": med([x[v]["surprise_left"] for x in rs]),
                  "median_anti_image_score": med([x[v]["anti_image_score"] for x in rs]),
                  "median_astra_reconstruction_score": med([x[v]["astra_reconstruction_score"] for x in rs])}
        if three:
            out[v]["own_preferred"] = int(sum(x[v]["own_preferred"] for x in rs))
            out[v]["median_surprise_left_unexpected"] = med([x[v]["surprise_left_unexpected"] for x in rs])
    out["bodies_with_surprise_below_100pct"] = int(sum(x["trained"]["surprise_left"] < 1 for x in rs))
    out["max_abs_weight_median"] = med([x["max_abs_weight"] for x in rs])
    out["clipped_frames_training_and_probe"] = int(sum(x["training_clipped_frames"] + x["trained"]["probe_clipped"] for x in rs))
    return out

def paired(rs, control, key="surprise_left"):
    """control minus trained, per schedule mean; positive = trained is better (less surprise left)."""
    seeds = sorted({x["seed"] for x in rs})
    d = np.array([np.mean([x[control][key] for x in rs if x["seed"] == s]) - np.mean([x["trained"][key] for x in rs if x["seed"] == s]) for s in seeds])
    rng = np.random.default_rng(890)
    boot = d[rng.integers(0, len(d), (20000, len(d)))].mean(axis=1)
    return {"mean_difference": float(d.mean()), "paired_seed_bootstrap_95": np.quantile(boot, [.025, .975]).tolist()}

S = {"three_jittered_PRIMARY": block(sel("three", True), True), "three_regular": block(sel("three", False), True),
     "two_jittered": block(sel("two", True), False), "two_regular": block(sel("two", False), False),
     "amplifying_control_three_jittered": block(sel("three", True, -1.0), True)}
P = sel("three", True)
S["primary_contrasts_control_minus_trained_surprise_left"] = {c: paired(P, c) for c in ("destinations_shuffled", "timers_shuffled", "untrained")}
by_order = {o: {"median_surprise_left": med([x["trained"]["surprise_left"] for x in P if x["order"] == o]),
                "median_anti_image_score": med([x["trained"]["anti_image_score"] for x in P if x["order"] == o])} for o in ("ABC", "ACB")}
S["primary_by_order"] = by_order
t = S["three_jittered_PRIMARY"]["trained"]; c = S["primary_contrasts_control_minus_trained_surprise_left"]
S["decision_rule"] = {
    "1_median_surprise_left_at_most_0.75": t["median_surprise_left"] <= 0.75,
    "2_median_anti_image_score_at_least_0.10": t["median_anti_image_score"] >= 0.10,
    "3_own_preferred_at_least_18_of_24": t["own_preferred"] >= 18,
    "4_beats_both_shuffles_bootstrap": bool(c["destinations_shuffled"]["paired_seed_bootstrap_95"][0] > 0 and c["timers_shuffled"]["paired_seed_bootstrap_95"][0] > 0),
    "astra_original_criterion_reconstruction_above_0.10": t["median_astra_reconstruction_score"] > 0.10}
S["decision_rule"]["all_four_pass"] = all(S["decision_rule"][k] for k in list(S["decision_rule"])[:4])
two = sel("two", True) + sel("two", False)
ch = [(x["chosen_timer"]["b_onset_tick"], x["chosen_timer"]["a_only_first_spike_tick"], x["chosen_timer"]["decay"], x["jitter"]) for x in two]
lag = [a - b for b, a, _, _ in ch if a is not None]
S["chosen_timer_two_event"] = {"bodies": len(ch), "spike_minus_b_onset_ticks": lag, "median_abs_offset": med(np.abs(lag)),
                               "correlation_spike_time_vs_b_onset": float(np.corrcoef([b for b, a, _, _ in ch if a is not None], [a for b, a, _, _ in ch if a is not None])[0, 1])}
(HERE / "summary_damping.json").write_text(json.dumps(S, indent=1)); print(json.dumps(S, indent=1))

# ---------------- figure ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, ACC, ACC2 = "#1f2a33", "#7d8b97", "#c8552d", "#2a6f97"
fig = plt.figure(figsize=(12.5, 9.2), layout="constrained"); gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.1, 1.1])
ax = fig.add_subplot(gs[0, 0])
labels = ["Untrained (passive)", "Timers shuffled", "Destinations shuffled", "Trained: 3 events, jittered", "Trained: 3 events, regular", "Trained: 2 events, jittered", "Trained: 2 events, regular"]
vals = [100 * S["three_jittered_PRIMARY"][k]["median_surprise_left"] for k in ("untrained", "timers_shuffled", "destinations_shuffled", "trained")] + \
       [100 * S[k]["trained"]["median_surprise_left"] for k in ("three_regular", "two_jittered", "two_regular")]
bars = ax.barh(labels, vals, color=[MUTED] * 3 + [ACC] * 4, height=.62); ax.invert_yaxis()
ax.axvline(100, color=INK, ls="--", lw=1.2); ax.bar_label(bars, fmt="%.0f%%", padding=4, color=INK)
ax.set_xlim(0, max(vals) * 1.18); ax.set_xlabel("Median surprise left when the learned sequence arrives (100% = no prediction)")
ax.set_title("Expected input: how much of the wave is still there", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[0, 1])
b = [x[0] for x in ch if x[1] is not None]; a = [x[1] for x in ch if x[1] is not None]; j = [x[3] for x in ch if x[1] is not None]
ax.plot([min(b) - 2, max(b) + 2], [min(b) - 2, max(b) + 2], color=MUTED, lw=1, ls="--", label="fires exactly when B starts")
ax.scatter([x for x, k in zip(b, j) if k], [y for y, k in zip(a, j) if k], color=ACC, s=38, label="jittered training", zorder=3)
ax.scatter([x for x, k in zip(b, j) if not k], [y for y, k in zip(a, j) if not k], facecolors="none", edgecolors=ACC2, s=60, label="regular training", zorder=3)
ax.set_xlabel("Frame at which B starts"); ax.set_ylabel("Frame at which the chosen timer fires (A alone)")
ax.set_title("Which fade the rule picks, two-event bodies", loc="left", fontweight="bold"); ax.legend(frameon=False, fontsize=9)
for row, order in enumerate(("ABC", "ACB")):
    d = np.load(HERE / "traces" / ("130001_three_jitter_%s.npz" % order))
    base, T = d["base"], d["target"]; t = np.arange(20, 240)
    for col, (name, wave, ttl) in enumerate((("expected", d["expected"], "the full sequence arrives"), ("cue", d["cue"], "only A arrives"))):
        ax = fig.add_subplot(gs[1 + row, col])
        for site, color, lab in ((7, ACC2, "B"), (4, ACC, "C")):
            ax.plot(t, T[t, site], color=color, lw=1.1, alpha=.45, label="%s: missing wave if nothing predicted" % lab)
            ax.plot(t, (wave - base)[t, site], color=color, lw=2.0, label="%s: what the trained body leaves" % lab)
        ax.axhline(0, color=MUTED, lw=.8)
        ax.set_title("Learned %s; %s" % (order, ttl), loc="left", fontsize=11)
        ax.set_xlabel("Simulation frame"); ax.set_ylabel("Field at sensor, beyond the A wave")
        if row == 0 and col == 0:
            ax.legend(fontsize=7.5, frameon=False, ncol=2)
fig.savefig(HERE / "damping_results.png", dpi=150, bbox_inches="tight"); plt.close(fig)
