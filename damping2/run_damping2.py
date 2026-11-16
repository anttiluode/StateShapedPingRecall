"""Predictive damping 2: does fixing the timer bank make the second event predictable? See PROTOCOL_DAMPING2.md.

Two factors, crossed, on the primary task of damping/ (three events, jittered training gaps):
  spacing  bunched : the original 24 fade rates (16 of 24 timers fire within ticks 51-61 of the A cue)
           even    : 24 fade rates chosen so the timers fire about 4 ticks apart over the reachable ticks 51-150
  input    shared  : timers listen to the shared field, which carries the body's own anti-waves (original)
           world   : timers listen to the world's wave only (the passive field of the stimuli at the same
                     aperture); the body's own writes cannot move its clock. Branches still learn from the
                     shared field at their tips.
The learning rule, its settings, stimuli, probe, scoring and controls are those of damping/ unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "damping"))
import engine as E
import run_damping as R1
import linear_bound as LB

BUNCHED = R1.DECAYS
EVEN = [0.756, 0.8405, 0.891, 0.9095, 0.921, 0.9225, 0.923, 0.925, 0.928, 0.933, 0.9405, 0.9455,
        0.949, 0.9515, 0.956, 0.961, 0.9615, 0.962, 0.9625, 0.9635, 0.964, 0.965, 0.9665, 0.9695]
RULE, EPISODES, STEPS, ONSET = R1.RULE, R1.EPISODES, R1.STEPS, R1.PROBE_ONSET
SEEDS = list(range(140001, 140013))          # fresh: not used in development or in damping/
CELLS = [("bunched", "shared"), ("even", "shared"), ("bunched", "world"), ("even", "world")]
_TH, _G = {}, None


def TH(decays):
    k = tuple(decays)
    if k not in _TH:
        _TH[k] = E.timer_thresholds(decays)
    return _TH[k]


def run(decays, W, stimuli, steps, learn=False, rule=None, world=False):
    """engine.run, with the option that timers hear only the world's wave (not the body's own writes)."""
    if not world:
        return E.run(decays, TH(decays), W, stimuli, steps, learn=learn, rule=rule)
    heard = E.passive(stimuli, steps)[:, E.TIMER_READ]
    rule = rule or {}
    field = E.make("WaveFieldNode", E.WAVE)
    units = [E.make("FadingHistorySpikeNode", {"decay": d, "arm_threshold": th[0], "release_threshold": th[1],
                                               "pulse_amplitude": E.PING_AMP, "refractory": 3, "enabled": 1})
             for d, th in zip(decays, TH(decays))]
    grows = [E.make("DampingReturnNode", dict(rule, weights=list(w), learn_enabled=int(learn))) for w in W]
    nodes = [field] + units + grows
    wave, spikes, clip = [], [[] for _ in units], 0
    for tick in range(steps):
        prev_read = [field.get_output("read_%d" % s) for s in range(E.N_SITES)]
        prev_pulse = [u.get_output("pulse") for u in units]
        prev_out = [[g.get_output("out_%d" % s) for s in range(E.N_SITES)] for g in grows]
        for n in nodes:
            n.pre_step()
        for u in units:
            u.set_input("signal_in", float(heard[tick - 1]) if tick > 0 else 0.0)   # same one-tick lag as engine.run
        for g, p, o in zip(grows, prev_pulse, prev_out):
            g.set_input("pre_spike", p)
            for s in range(E.N_SITES):
                g.set_input("site_%d" % s, prev_read[s])
                if o[s]:
                    field.set_input("write_%d" % s, o[s])
        for site, onset, scale in stimuli:
            if onset + 1 <= tick < onset + 3:
                field.set_input("write_%d" % site, E.STIM_AMP * scale)
        for n in nodes:
            n.step()
        for k, u in enumerate(units):
            if u.pulse:
                spikes[k].append(tick)
        wave.append(field.read_values.copy())
        clip += int(float(np.max(np.abs(field.field))) >= 3.9998 or float(np.max(np.abs(field.vel))) >= 3.9998)
    return {"wave": np.asarray(wave, dtype=float), "weights": [list(map(float, g.weights)) for g in grows],
            "spikes": spikes, "clipped": clip}


def measure(decays, world, W, order, gap, other):
    a, b = ONSET + gap + 2, ONSET + (len(order) - 1) * gap + 100
    m2 = ONSET + 2 * gap + 2                        # start of the second-event window
    base = E.passive(R1.seq("A", ONSET, gap), STEPS)
    T = E.passive(R1.seq(order, ONSET, gap), STEPS) - base
    To = E.passive(R1.seq(other, ONSET, gap), STEPS) - base
    den, den2 = float((T[a:b] ** 2).sum()), float((T[m2:b] ** 2).sum())
    full = run(decays, W, R1.seq(order, ONSET, gap), STEPS, world=world)
    cue = run(decays, W, R1.seq("A", ONSET, gap), STEPS, world=world)
    unexp = run(decays, W, R1.seq(other, ONSET, gap), STEPS, world=world)
    p, left = cue["wave"] - base, full["wave"] - base
    return {"surprise_left": float((left[a:b] ** 2).sum()) / den,
            "surprise_left_first_event_window": float((left[a:m2] ** 2).sum()) / float((T[a:m2] ** 2).sum()),
            "surprise_left_second_event_window": float((left[m2:b] ** 2).sum()) / den2,
            "astra_reconstruction_score": 1 - float(((p - T)[a:b] ** 2).sum()) / den,
            "anti_image_score": 1 - float(((p + T)[a:b] ** 2).sum()) / den,
            "own_preferred": bool(((p + T)[a:b] ** 2).sum() < ((p + To)[a:b] ** 2).sum() - 1e-12),
            "surprise_left_unexpected": float(((unexp["wave"] - base)[a:b] ** 2).sum()) / float((To[a:b] ** 2).sum()),
            "probe_clipped": full["clipped"] + cue["clipped"] + unexp["clipped"],
            "cue_spikes": [list(s) for s in cue["spikes"]], "window": [a, m2, b], "_T": T}


def bounds(cue_spikes, W, T, window):
    """Post-hoc style linear bounds, computed here for every body (see damping/linear_bound.py)."""
    global _G
    if _G is None:
        _G = LB.impulse_responses()
    if not all(len(s) == 1 for s in cue_spikes):
        return {"exact": False}
    a, _, b = window
    B = LB.basis(_G, [s[0] for s in cue_spikes])
    y, den, n = -T[a:b].ravel(), float((T[a:b] ** 2).sum()), len(cue_spikes)
    X = B[:, :, a:b, :].reshape(n * E.N_SITES, -1).T
    Xb = X.reshape(X.shape[0], n, E.N_SITES).sum(axis=1)
    wc, wb = np.linalg.lstsq(X, y, rcond=None)[0], np.linalg.lstsq(Xb, y, rcond=None)[0]
    local = 1 - float(((X @ np.asarray(W).ravel() - y) ** 2).sum()) / den
    return {"exact": True, "local_rule_linear": local,
            "timing_blind_8": 1 - float(((Xb @ wb - y) ** 2).sum()) / den,
            "ceiling_192": 1 - float(((X @ wc - y) ** 2).sum()) / den, "ceiling_weight_max_abs": float(np.abs(wc).max())}


def job(arg):
    seed, spacing, inp = arg
    decays, world = (BUNCHED if spacing == "bunched" else EVEN), inp == "world"
    gap, amp, eps = R1.schedule(seed, True)
    rows = []
    for order in ("ABC", "ACB"):
        other = "ACB" if order == "ABC" else "ABC"
        W = [[0.0] * 8 for _ in decays]; clip = 0
        for onset, g in eps:
            r = run(decays, W, R1.seq(order, onset, g, amp), STEPS, learn=True, rule=dict(RULE, sign=1.0), world=world)
            W, clip = r["weights"], clip + r["clipped"]
        rng = np.random.default_rng(seed + 7000)
        variants = {"trained": W, "destinations_shuffled": [list(np.asarray(w)[rng.permutation(8)]) for w in W],
                    "timers_shuffled": [W[i] for i in rng.permutation(len(W))],
                    "untrained": [[0.0] * 8 for _ in decays]}
        rec = {"seed": seed, "spacing": spacing, "input": inp, "order": order, "gap": gap, "amplitude": amp,
               "training_clipped_frames": clip, "weights": W, "max_abs_weight": float(np.abs(W).max())}
        for name, Wv in variants.items():
            m = measure(decays, world, Wv, order, gap, other)
            T = m.pop("_T")
            if name == "trained":
                rec["bounds"] = bounds(m["cue_spikes"], Wv, T, m["window"])
                rec["timers_moved_by_own_weights"] = None
            rec[name] = m
        passive_cue = rec["untrained"]["cue_spikes"]
        rec["timers_moved_by_own_weights"] = int(sum(x != y for x, y in zip(rec["trained"]["cue_spikes"], passive_cue)))
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    seeds = [991] if a.smoke else SEEDS
    jobs = [(s, sp, inp) for sp, inp in CELLS for s in seeds]
    receipt = {"label": "smoke plumbing only" if a.smoke else "predictive damping 2, 2x2 batch",
               "cells": CELLS, "bunched_decays": BUNCHED, "even_decays": EVEN, "rule": RULE, "episodes": EPISODES,
               "seeds": seeds, "results": [],
               "protocol_sha256": hashlib.sha256((HERE / "PROTOCOL_DAMPING2.md").read_bytes()).hexdigest(),
               "runner_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
               "engine_sha256": hashlib.sha256((HERE.parent / "damping" / "engine.py").read_bytes()).hexdigest(),
               "node_sha256": hashlib.sha256((HERE.parent / "damping" / "damping_return_node.py").read_bytes()).hexdigest()}
    out = HERE / ("smoke_damping2.json" if a.smoke else "results_damping2.json")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for n, rows in enumerate(pool.map(job, jobs), 1):
            receipt["results"] += rows; receipt["elapsed_seconds"] = time.time() - t0
            out.write_text(json.dumps(receipt, indent=1))
            print("%d/%d jobs  %.0fs" % (n, len(jobs), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
