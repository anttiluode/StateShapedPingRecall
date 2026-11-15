"""Predictive-damping batch; see PROTOCOL_DAMPING.md. Needs numpy and opencv-python-headless."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine as E

DECAYS = [round(0.78 + 0.008 * i, 3) for i in range(24)]
RULE = {"learning_rate": 0.03, "listen_from": 3, "listen_to": 5, "self_response": 0.71, "max_weight": 2.0}
EPISODES, STEPS, PROBE_ONSET = 20, 360, 23
SEEDS = list(range(130001, 130013))
TRACE_SEED = 130001                 # traces saved for the first seed, chosen in advance
_TH = None
def TH():
    global _TH
    if _TH is None:
        _TH = E.timer_thresholds(DECAYS)
    return _TH


def seq(order, onset, gap, amp=1.0):
    return [(E.SITES[c], onset + i * gap, amp) for i, c in enumerate(order)]


def schedule(seed, jitter):
    """Astra's generator (run.py: job), with the training-gap jitter optionally switched off."""
    rng = np.random.default_rng(seed)
    gap = int(rng.integers(28, 43)); amp = float(rng.uniform(.9, 1.12))
    eps = [(20 + int(rng.integers(-5, 6)), gap + int(rng.integers(-3, 4))) for _ in range(EPISODES)]
    if not jitter:
        eps = [(on, gap) for on, _ in eps]
    return gap, amp, eps


def train(order, eps, amp, sign=1.0):
    W = [[0.0] * 8 for _ in DECAYS]; clip = 0; spikes = 0
    for onset, g in eps:
        r = E.run(DECAYS, TH(), W, seq(order, onset, g, amp), STEPS, learn=True, rule=dict(RULE, sign=sign))
        W = r["weights"]; clip += r["clipped"]; spikes += sum(len(s) for s in r["spikes"])
    return W, clip, spikes


def measure(W, order, gap, other=None, keep=False):
    win = (PROBE_ONSET + gap + 2, PROBE_ONSET + (len(order) - 1) * gap + 100)
    a, b = win
    base = E.passive(seq("A", PROBE_ONSET, gap), STEPS)
    T = E.passive(seq(order, PROBE_ONSET, gap), STEPS) - base
    den = float((T[a:b] ** 2).sum())
    full = E.run(DECAYS, TH(), W, seq(order, PROBE_ONSET, gap), STEPS)
    cue = E.run(DECAYS, TH(), W, seq("A", PROBE_ONSET, gap), STEPS)
    p = cue["wave"] - base
    out = {"surprise_left": float(((full["wave"] - base)[a:b] ** 2).sum()) / den,
           "astra_reconstruction_score": 1 - float(((p - T)[a:b] ** 2).sum()) / den,
           "anti_image_score": 1 - float(((p + T)[a:b] ** 2).sum()) / den,
           "probe_clipped": full["clipped"] + cue["clipped"], "cue_spikes": sum(len(s) for s in cue["spikes"])}
    if other:
        To = E.passive(seq(other, PROBE_ONSET, gap), STEPS) - base
        out["own_preferred"] = bool(((p + T)[a:b] ** 2).sum() < ((p + To)[a:b] ** 2).sum() - 1e-12)
        unexp = E.run(DECAYS, TH(), W, seq(other, PROBE_ONSET, gap), STEPS)
        out["surprise_left_unexpected"] = float(((unexp["wave"] - base)[a:b] ** 2).sum()) / float((To[a:b] ** 2).sum())
    if keep:
        out["_traces"] = {"base": base, "target": T, "cue": cue["wave"], "expected": full["wave"]}
    return out


def job(arg):
    seed, task, jitter, sign = arg
    gap, amp, eps = schedule(seed, jitter)
    rows = []
    orders = ("ABC", "ACB") if task == "three" else ("AB",)
    for order in orders:
        other = {"ABC": "ACB", "ACB": "ABC"}.get(order)
        W, clip, spikes = train(order, eps, amp, sign)
        keep = seed == TRACE_SEED and sign > 0
        m = measure(W, order, gap, other, keep)
        tr = m.pop("_traces", None)
        rng = np.random.default_rng(seed + 7000)
        Wd = [list(np.asarray(w)[rng.permutation(8)]) for w in W]
        Wt = [W[i] for i in rng.permutation(len(W))]
        rec = {"seed": seed, "task": task, "jitter": bool(jitter), "sign": sign, "order": order, "gap": gap, "amplitude": amp,
               "training_clipped_frames": clip, "training_spikes": spikes, "weights": W,
               "max_abs_weight": float(np.abs(W).max()), "trained": m}
        if sign > 0:
            rec["destinations_shuffled"] = measure(Wd, order, gap, other)
            rec["timers_shuffled"] = measure(Wt, order, gap, other)
            rec["untrained"] = measure([[0.0] * 8 for _ in DECAYS], order, gap, other)
        if task == "two":
            Wa = np.abs(np.asarray(W))[:, E.SITES["B"]]
            k = int(Wa.argmax())
            solo = E.run(DECAYS, TH(), [[0.0] * 8 for _ in DECAYS], seq("A", PROBE_ONSET, gap), STEPS)["spikes"]
            rec["chosen_timer"] = {"decay": DECAYS[k], "weight_at_B": float(np.asarray(W)[k, E.SITES["B"]]),
                                   "a_only_first_spike_tick": (solo[k][0] if solo[k] else None),
                                   "b_onset_tick": PROBE_ONSET + gap}
        if tr is not None:
            d = HERE / "traces"; d.mkdir(exist_ok=True)
            np.savez_compressed(d / ("%d_%s_%s_%s.npz" % (seed, task, "jitter" if jitter else "regular", order)), **tr)
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    seeds = [991] if a.smoke else SEEDS
    jobs = ([(s, "three", True, 1.0) for s in seeds] + [(s, "three", False, 1.0) for s in seeds] +
            [(s, "two", True, 1.0) for s in seeds] + [(s, "two", False, 1.0) for s in seeds] +
            [(s, "three", True, -1.0) for s in seeds])
    receipt = {"label": "smoke plumbing only" if a.smoke else "predictive damping, 12-schedule batch",
               "decays": DECAYS, "rule": RULE, "episodes": EPISODES, "steps": STEPS, "seeds": seeds,
               "thresholds": TH(), "results": [],
               "protocol_sha256": hashlib.sha256((HERE / "PROTOCOL_DAMPING.md").read_bytes()).hexdigest(),
               "runner_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
               "engine_sha256": hashlib.sha256((HERE / "engine.py").read_bytes()).hexdigest(),
               "node_sha256": hashlib.sha256((HERE / "damping_return_node.py").read_bytes()).hexdigest()}
    out = HERE / ("smoke_damping.json" if a.smoke else "results_damping.json")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for n, rows in enumerate(pool.map(job, jobs), 1):
            receipt["results"] += rows; receipt["elapsed_seconds"] = time.time() - t0
            out.write_text(json.dumps(receipt, indent=1))
            print("%d/%d jobs  %.0fs" % (n, len(jobs), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
