"""Engine for the damping experiments: real upstream WaveFieldNode and FadingHistorySpikeNode in host step order.

A BODY is a bank of fade-triggered timer units that all listen near stimulus site A, each with its own
fade rate, each returning to the field through a DampingReturnNode (eight signed branches).
Fast state is rebuilt from nothing every episode; only the branch weights carry over.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "tests"))
from host_shim import load

N_SITES = 8
SITES = {"A": 0, "B": 7, "C": 4}
TIMER_READ = 1                       # all timers listen at aperture 1, next to A
STIM_AMP, PING_AMP = 0.12, 0.24      # as in PerceptionLabAI 004-010
WAVE = {"size": 96, "speed": 0.5, "damping": 0.006, "substeps": 4, "drive_gain": 0.6, "source_radius": 4,
        "edge_absorb": 0.06, "read_aperture": 6, "line_y": 0.5, "first_x": 0.2, "last_x": 0.92, "source_x": 0.08}

_cls = {}
def cls(name):
    if name not in _cls:
        path = {"WaveFieldNode": ROOT / "vendor/up/nodes/wavefieldnode.py",
                "FadingHistorySpikeNode": ROOT / "vendor/nodes/fading_history_spike_node.py",
                "DampingReturnNode": ROOT / "damping/damping_return_node.py"}[name]
        _cls[name] = load(path, name)
    return _cls[name]


def make(name, cfg):
    n = cls(name)()
    if hasattr(n, "set_config_options"):
        n.set_config_options(dict(cfg))
    for k, v in cfg.items():
        if hasattr(n, k) and getattr(n, k) != v:
            setattr(n, k, v)
    n._render = lambda *a: None        # drawing only
    return n


def run(decays, thresholds, weights, stimuli, steps, learn=False, rule=None, record_units=False):
    """stimuli: list of (site, onset_frame, amplitude_scale); each is a two-frame pulse like PulseSourceNode width 2.
    Returns sensor waves (steps x 8), final weights, timer spike times, clipped frames."""
    rule = rule or {}
    field = make("WaveFieldNode", WAVE)
    assert field.field.shape == (96, 96)
    units = [make("FadingHistorySpikeNode", {"decay": d, "arm_threshold": th[0], "release_threshold": th[1],
                                             "pulse_amplitude": PING_AMP, "refractory": 3, "enabled": 1})
             for d, th in zip(decays, thresholds)]
    grows = [make("DampingReturnNode", dict(rule, weights=list(w), learn_enabled=int(learn))) for w in weights]
    nodes = [field] + units + grows
    wave, spikes, clip = [], [[] for _ in units], 0
    for tick in range(steps):
        prev_read = [field.get_output("read_%d" % s) for s in range(N_SITES)]
        prev_pulse = [u.get_output("pulse") for u in units]
        prev_out = [[g.get_output("out_%d" % s) for s in range(N_SITES)] for g in grows]
        for n in nodes:
            n.pre_step()
        for u in units:
            u.set_input("signal_in", prev_read[TIMER_READ])
        for g, p, o in zip(grows, prev_pulse, prev_out):
            g.set_input("pre_spike", p)
            for s in range(N_SITES):
                g.set_input("site_%d" % s, prev_read[s])
                if o[s]:
                    field.set_input("write_%d" % s, o[s])
        for site, onset, scale in stimuli:
            if onset + 1 <= tick < onset + 3:
                field.set_input("write_%d" % site, STIM_AMP * scale)
        for n in nodes:
            n.step()
        for k, u in enumerate(units):
            if u.pulse:
                spikes[k].append(tick)
        wave.append(field.read_values.copy())
        clip += int(float(np.max(np.abs(field.field))) >= 3.9998 or float(np.max(np.abs(field.vel))) >= 3.9998)
    return {"wave": np.asarray(wave, dtype=float), "weights": [list(map(float, g.weights)) for g in grows],
            "spikes": spikes, "clipped": clip, "peak_trace": None}


def passive(stimuli, steps):
    return run([], [], [], stimuli, steps)["wave"]


def timer_thresholds(decays, steps=200, arm=0.80, release=0.20):
    """Label-blind: each timer's arm/release from its own peak trace under one passive A pulse.

    arm = 0.80 of that peak means a timer is started only by a strong, nearby event (A, 10 px away).
    With the 0.45 used in PerceptionLabAI 009 the timers were re-started by the later events' waves
    and three-event sequences failed in development; see README."""
    w = passive([(SITES["A"], 20, 1.0)], steps)[:, TIMER_READ]
    out = []
    for d in decays:
        tr, peak = 0.0, 0.0
        for v in w:
            tr = d * tr + (1 - d) * abs(v); peak = max(peak, tr)
        out.append((arm * peak, release * peak))
    return out
