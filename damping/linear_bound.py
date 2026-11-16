"""Post-hoc diagnostic for the damping batch (written AFTER its results; not part of the decision rule).

With growth frozen, each timer fires once on the A-only cue and the wave field is linear, so the
cue response is a weighted sum of fixed impulse responses:

    p(t, sensor) = sum_i sum_s  W[i, s] * G_s(t - t_i - 2, sensor)

where t_i is timer i's firing tick (the shift is t_i + 2 in host order) and G_s the field's response at the eight sensors to one ping at site s.
That makes three things exactly computable for every body:

  ceiling      best possible anti-image score with these firing times (192 free weights, least squares)
  timing-blind best score if every timer must carry the SAME eight weights (8 free weights)
  local rule   the score the trained body actually got (from the batch receipt)

and it checks the linearity assumption by comparing the predicted response of the trained weights
with a direct simulation.

One complication found while writing this: the trained body's own anti-waves reach the timers' aperture
and move the slow timers' firing (some by -19 ticks, some by +40 to +80). So everything is computed twice:
with the PASSIVE firing times (what the timers do with zero weights) and with the body's OWN firing times
under its trained weights. The linearity check uses the body's own times. Both ceilings assume the firing
times stay fixed while the weights change, which a real closed loop does not guarantee: they are bounds
for this read-out geometry, not achievable targets.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine as E
import run_damping as R

STEPS, ONSET = R.STEPS, R.PROBE_ONSET


def impulse_responses():
    """G[s]: (STEPS x 8) sensor response to one ping written at site s on tick 0, in host step order."""
    G = []
    for s in range(E.N_SITES):
        f = E.make("WaveFieldNode", E.WAVE)
        rows = []
        for tick in range(STEPS):
            f.pre_step()
            if tick == 0:
                f.set_input("write_%d" % s, E.PING_AMP)
            f.step()
            rows.append(f.read_values.copy())
        G.append(np.asarray(rows, dtype=float))
    return G


def basis(G, times):
    """B[i, s]: (STEPS x 8) contribution of weight W[i,s] = 1, for timer i firing at times[i]."""
    B = np.zeros((len(times), E.N_SITES, STEPS, E.N_SITES))
    for i, t in enumerate(times):
        start = t + 2                     # pulse on tick t -> return node sees it on t+1 -> field written on t+2
        for s in range(E.N_SITES):
            B[i, s, start:] = G[s][:STEPS - start]
    return B


def main():
    rows = json.loads((HERE / "results_damping.json").read_text())["results"]
    G = impulse_responses()
    zero = [[0.0] * 8 for _ in R.DECAYS]
    cue0 = E.run(R.DECAYS, R.TH(), zero, R.seq("A", ONSET, 30), STEPS)
    times = [s[0] for s in cue0["spikes"]]
    assert all(len(s) == 1 for s in cue0["spikes"]), "every timer must fire exactly once on the cue"
    B0 = basis(G, times)
    rng = np.random.default_rng(1)
    perm_moves = [float(np.median(np.abs(np.asarray(times) - np.asarray(times)[rng.permutation(len(times))])))
                  for _ in range(2000)]
    out = {"timer_fire_ticks_on_A_cue_passive": times,
           "timers_within_10_ticks_of_tick_51": int(sum(51 <= t <= 61 for t in times)),
           "median_timing_change_under_a_random_timer_shuffle_ticks": float(np.median(perm_moves)),
           "bodies": []}

    def scores(B, y, den, a, b, n):
        X = B[:, :, a:b, :].reshape(n * E.N_SITES, -1).T
        wc = np.linalg.lstsq(X, y, rcond=None)[0]
        Xb = X.reshape(X.shape[0], n, E.N_SITES).sum(axis=1)
        wb = np.linalg.lstsq(Xb, y, rcond=None)[0]
        return (1 - float(((X @ wc - y) ** 2).sum()) / den, 1 - float(((Xb @ wb - y) ** 2).sum()) / den,
                float(np.abs(wc).max()))
    for r in rows:
        if r["sign"] < 0:
            continue
        a, b = ONSET + r["gap"] + 2, ONSET + (len(r["order"]) - 1) * r["gap"] + 100
        base = E.passive(R.seq("A", ONSET, r["gap"]), STEPS)
        T = E.passive(R.seq(r["order"], ONSET, r["gap"]), STEPS) - base
        y = -T[a:b].ravel()                                    # the anti-image target
        den = float((T[a:b] ** 2).sum())
        W = np.asarray(r["weights"])
        sim = E.run(R.DECAYS, R.TH(), r["weights"], R.seq("A", ONSET, r["gap"]), STEPS)
        own = [sp[0] for sp in sim["spikes"]]
        single = all(len(sp) == 1 for sp in sim["spikes"])
        Bown = basis(G, own)
        # 1) linearity check: trained weights through the body's OWN firing times vs direct simulation
        lin = np.tensordot(W, Bown, axes=([0, 1], [0, 1]))
        lin_err = float(np.abs((sim["wave"] - base) - lin).max() / (np.abs(sim["wave"] - base).max() + 1e-12))
        # 2) ceilings (192 free weights) and 3) timing-blind (one shared row of 8), passive and own times
        ceil0, blind0, wmax0 = scores(B0, y, den, a, b, len(times))
        ceil, blind, wmax = scores(Bown, y, den, a, b, len(own))
        # 4) how much of the trained weights is one shared row?
        common = float((W.mean(axis=0) ** 2).sum() * len(W) / ((W ** 2).sum() + 1e-12))
        out["bodies"].append({"seed": r["seed"], "task": r["task"], "jitter": r["jitter"], "order": r["order"],
                              "local_rule": r["trained"]["anti_image_score"],
                              "timers_shuffled": r["timers_shuffled"]["anti_image_score"],
                              "ceiling_192_own_times": ceil, "timing_blind_8_own_times": blind,
                              "ceiling_192_passive_times": ceil0, "timing_blind_8_passive_times": blind0,
                              "ceiling_weight_max_abs_own_times": wmax, "shared_row_energy_fraction": common,
                              "own_fire_ticks": own, "each_timer_fired_once": single,
                              "slow_timers_moved_by_own_weights": int(sum(o != t for o, t in zip(own, times))),
                              "linearity_rel_error": lin_err})
    med = lambda k, sel: float(np.median([x[k] for x in out["bodies"] if sel(x)]))
    groups = {"three_jittered_PRIMARY": lambda x: x["task"] == "three" and x["jitter"],
              "three_regular": lambda x: x["task"] == "three" and not x["jitter"],
              "two_jittered": lambda x: x["task"] == "two" and x["jitter"],
              "two_regular": lambda x: x["task"] == "two" and not x["jitter"]}
    keys = ("local_rule", "timers_shuffled", "timing_blind_8_own_times", "ceiling_192_own_times",
            "timing_blind_8_passive_times", "ceiling_192_passive_times", "ceiling_weight_max_abs_own_times",
            "shared_row_energy_fraction", "slow_timers_moved_by_own_weights", "linearity_rel_error")
    out["summary"] = {g: {k: med(k, f) for k in keys} for g, f in groups.items()}
    out["all_bodies_each_timer_fired_once"] = bool(all(x["each_timer_fired_once"] for x in out["bodies"]))
    (HERE / "linear_bound.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))
    print({k: v for k, v in out.items() if k not in ("bodies", "summary")})


if __name__ == "__main__":
    main()
