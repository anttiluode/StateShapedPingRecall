"""009 core: build the growing-body graph and step it in host order with the real upstream nodes."""
from __future__ import annotations
import copy, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from host_shim import load

N_SITES = 8
STIM_SITE = {"A": 0, "B": 7}
# id layout: 0,1 stimulus sources; 2 wave; 3.. units; 20.. growers; 40 probe source
UNIT_READS = (1, 4, 6)
UNIT_DECAYS = (0.90, 0.93, 0.90)
UNIT_IDS = tuple(3 + k for k in range(len(UNIT_READS)))
GROW_IDS = tuple(20 + k for k in range(len(UNIT_READS)))
PROBE_ID = 40
FILES = {"PulseSourceNode": ("up", "pulsesourcenode.py"), "WaveFieldNode": ("up", "wavefieldnode.py"),
         "FadingHistorySpikeNode": ("ws", "fading_history_spike_node.py"),
         "GrowingReturnNode": ("ws", "growing_return_node.py")}


def edge(a, ap, b, bp, w=1):
    return {"from_node_id": a, "from_port": ap, "to_node_id": b, "to_port": bp, "learned_weight": w}


def uniform_structure(budget=1.0):
    return {"weights": [[budget / N_SITES] * N_SITES for _ in UNIT_IDS],
            "alive": [[1] * N_SITES for _ in UNIT_IDS]}


def build_graph(structure, thresholds, stim=None, probe_site=None, learn=1, params=None, stim_amp=0.12):
    """stim: list of (site_letter, frame, scale). probe_site: write index for a single neutral pulse at frame 20."""
    P = {"budget": 1.0, "learning_rate": 0.05, "trace_decay": 0.97, "prune_threshold": 0.02,
         "harden_rate": 0.0, "harden_mode": 1}
    P.update(params or {})
    nodes, edges = [], []
    srcs = {"A": 0, "B": 1}
    present = {s: (f, sc) for s, f, sc in (stim or [])}
    for letter, nid in srcs.items():
        f, sc = present.get(letter, (20, 0.0))
        nodes.append({"id": nid, "class_name": "PulseSourceNode", "pos_x": -720.0, "pos_y": -200.0 + 170 * nid,
                      "width": 205.0, "height": 140.0,
                      "config": {"node_title": "Stimulus %s" % letter, "period": 0, "width": 2,
                                 "amplitude": stim_amp * sc, "warmup": int(f)}, "custom_state_file": None})
        edges.append(edge(nid, "signal", 2, "write_%d" % STIM_SITE[letter]))
    nodes.append({"id": 2, "class_name": "WaveFieldNode", "pos_x": -440.0, "pos_y": -305.0, "width": 310.0, "height": 340.0,
                  "config": {"node_title": "One shared 96px wave", "size": 96, "speed": 0.5, "damping": 0.006,
                             "substeps": 4, "drive_gain": 0.6, "source_radius": 4, "edge_absorb": 0.06,
                             "read_aperture": 6, "line_y": 0.5, "first_x": 0.2, "last_x": 0.92, "source_x": 0.08},
                  "custom_state_file": None})
    for k, uid in enumerate(UNIT_IDS):
        th = thresholds[k]
        nodes.append({"id": uid, "class_name": "FadingHistorySpikeNode", "pos_x": -30.0, "pos_y": -310.0 + 200 * k,
                      "width": 325.0, "height": 180.0,
                      "config": {"node_title": "Unit %d / reads %d / fade %.2f" % (k, UNIT_READS[k], UNIT_DECAYS[k]),
                                 "decay": UNIT_DECAYS[k], "arm_threshold": th[0], "release_threshold": th[1],
                                 "pulse_amplitude": 0.24, "refractory": 3, "enabled": 1}, "custom_state_file": None})
        edges.append(edge(2, "read_%d" % UNIT_READS[k], uid, "signal_in"))
        gid = GROW_IDS[k]
        cfg = dict(P); cfg.update({"node_title": "Unit %d branches" % k, "learn_enabled": int(learn),
                                   "weights": list(structure["weights"][k]), "alive": list(structure["alive"][k])})
        if "hardness" in structure:
            cfg["hardness"] = list(structure["hardness"][k])
        nodes.append({"id": gid, "class_name": "GrowingReturnNode", "pos_x": 360.0, "pos_y": -310.0 + 200 * k,
                      "width": 325.0, "height": 170.0, "config": cfg, "custom_state_file": None})
        edges.append(edge(uid, "pulse", gid, "pre_spike"))
        for s in range(N_SITES):
            edges.append(edge(2, "read_%d" % s, gid, "site_%d" % s))
            edges.append(edge(gid, "out_%d" % s, 2, "write_%d" % s))
    if probe_site is not None:
        nodes.append({"id": PROBE_ID, "class_name": "PulseSourceNode", "pos_x": -720.0, "pos_y": 160.0,
                      "width": 205.0, "height": 140.0,
                      "config": {"node_title": "Neutral probe", "period": 0, "width": 2, "amplitude": stim_amp,
                                 "warmup": 20}, "custom_state_file": None})
        edges.append(edge(PROBE_ID, "signal", 2, "write_%d" % probe_site))
    return {"global_settings": {"coupling": 100, "speed": 100}, "nodes": nodes, "edges": edges}


def instantiate(graph, upstream):
    obj = {}
    for d in graph["nodes"]:
        where, fname = FILES[d["class_name"]]
        src = (pathlib.Path(upstream) / "nodes" if where == "up" else ROOT / "nodes") / fname
        n = load(src, d["class_name"])()
        cfg = copy.deepcopy(d.get("config", {}))
        if hasattr(n, "set_config_options"):
            n.set_config_options(cfg)
        for k, v in cfg.items():
            if hasattr(n, k) and getattr(n, k) != v:
                setattr(n, k, v)
        n._render = lambda *a: None       # drawing only
        obj[d["id"]] = n
    if obj[2].field.shape != (96, 96):
        raise RuntimeError("wave is not 96x96")
    for e in graph["edges"]:
        a, b = obj[e["from_node_id"]], obj[e["to_node_id"]]
        if e["from_port"] not in a.outputs or e["to_port"] not in b.inputs:
            raise RuntimeError("bad edge %s" % e)
    return obj


def simulate(graph, upstream, steps, snapshots=(), spike_decay=0.92, windows=None):
    """Returns final structure, unit spikes, per-site response energy in optional frame windows, and reader features."""
    obj = instantiate(graph, upstream)
    edges = graph["edges"]
    spikes = {u: [] for u in UNIT_IDS}
    ema = [0.0] * len(UNIT_IDS)
    feats, clip, peak_trace = {}, 0, [0.0] * len(UNIT_IDS)
    site_energy = {name: np.zeros(N_SITES) for name in (windows or {})}
    for tick in range(steps):
        for n in obj.values():
            n.pre_step()
        for e in edges:
            a = obj[e["from_node_id"]]
            v = a.get_output(e["from_port"])
            if v is not None:
                obj[e["to_node_id"]].set_input(e["to_port"], v, a.outputs[e["from_port"]], e.get("learned_weight", 1.0))
        for n in obj.values():
            n.step()
        for k, u in enumerate(UNIT_IDS):
            fired = 1.0 if obj[u].pulse else 0.0
            if fired:
                spikes[u].append(tick)
            ema[k] = spike_decay * ema[k] + (1 - spike_decay) * fired
            peak_trace[k] = max(peak_trace[k], float(obj[u].trace))
        for name, (lo, hi) in (windows or {}).items():
            if lo <= tick < hi:
                site_energy[name] += np.abs(obj[2].read_values.astype(float))
        clip += int(float(np.max(np.abs(obj[2].field))) >= 3.9998)
        if tick in snapshots:
            feats[tick] = [float(obj[u].trace) for u in UNIT_IDS] + list(ema)
    structure = {"weights": [list(map(float, obj[g].weights)) for g in GROW_IDS],
                 "alive": [list(map(int, obj[g].alive)) for g in GROW_IDS],
                 "hardness": [list(map(float, obj[g].hardness)) for g in GROW_IDS]}
    return {"structure": structure, "spikes": {str(u): spikes[u] for u in UNIT_IDS}, "features": feats,
            "clipped_frames": clip, "peak_trace": peak_trace,
            "site_energy": {k: [float(x) for x in v] for k, v in site_energy.items()}}


def calibrate(upstream):
    """Open loop (no returns): peak trace per unit from one A-then-B and one B-then-A pair. Label-blind."""
    zero = {"weights": [[0.0] * N_SITES for _ in UNIT_IDS], "alive": [[1] * N_SITES for _ in UNIT_IDS]}
    dummy = [(1.0, 0.5)] * len(UNIT_IDS)        # unreachable thresholds: units never fire in calibration
    peaks = [0.0] * len(UNIT_IDS)
    for stim in ([("A", 20, 1.0), ("B", 52, 1.0)], [("B", 20, 1.0), ("A", 52, 1.0)]):
        r = simulate(build_graph(zero, dummy, stim, learn=0), upstream, 240)
        peaks = [max(p, q) for p, q in zip(peaks, r["peak_trace"])]
    return {"peaks": peaks, "thresholds": [(0.45 * p, 0.20 * p) for p in peaks]}


def episode(structure, thresholds, upstream, order, gap, scale, steps=400, learn=1, params=None, jitter=0):
    first, second = (("A", "B") if order == "AB" else ("B", "A"))
    stim = [(first, 20 + jitter, scale), (second, 20 + jitter + gap, scale)]
    return simulate(build_graph(structure, thresholds, stim, learn=learn, params=params), upstream, steps)

