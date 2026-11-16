# Predictive damping 2: protocol

Written 16 November 2026 by Claude Opus 5.5, **before** the batch. Follow-up to [damping/](../damping/README.md).

## Question

In damping/, the local tip-flattening rule cancelled the first predicted event well and the second barely, and the timers-shuffled control cost too little. A post-hoc analysis suggested two causes in the **timer bank**, not the rule: 16 of 24 timers fire within ten ticks of each other, and a trained body's own anti-waves reach the timers and move the slow ones. If that diagnosis is right, fixing the bank should let the same rule predict the second event and use timing.

## Design

Two factors, crossed; everything else exactly as in damping/ (rule and settings, stimuli, schedule generator with jittered gaps, probe, scoring, shuffle controls, untrained baseline).

| Factor | Original | Changed |
|---|---|---|
| Spacing | the original 24 fade rates (bunched) | 24 fade rates chosen so that, passively, the timers fire on the A cue at ticks 51, 55, 60, 64, 68, 69, 86, 87, 88, 90, 94, 98, 103, 107, 111, 116, 117, 118, 141, 142, 143, 144, 146, 150 |
| Timer input | the shared field (hears the body's own writes) | the world's wave only: the passive field of the stimuli at the same aperture, with the same one-tick lag |

How the even rates were chosen, before any new results: scan fade rates 0.700–0.990 in steps of 0.0005, record each one's passive firing tick, then pick 24 as evenly as possible across the reachable ticks 51–150. Ticks 70–85 and 119–140 cannot be reached by this timer design (the fading trace has several lobes), so the spacing has those two holes. Thresholds are set label-blind as before (0.80 and 0.20 of each timer's passive A peak trace).

In the world-input condition the branches still learn from the shared field at their tips; only the timers' input changes. This is an architectural choice, not biology: it separates the clock from the body's output.

**Batch:** fresh schedule seeds 140001–140012 (not used in development or in damping/). Each seed trains one ABC and one ACB body in each of the four cells: 24 bodies per cell, 96 in all. The bunched/shared cell repeats the original design on new seeds.

**New measurements** in addition to those of damping/: surprise left split into the *first-event window* (from the first missing event's onset + 2 to the second missing event's onset + 2) and the *second-event window* (from the second missing event's onset + 2 to the end of the original window, 100 frames later); and, for every trained body whose timers each fire once on the cue, the exact linear bounds of `damping/linear_bound.py` (timing-blind and ceiling).

A plumbing run on development seed 991 (`--smoke`, not evidence) was made before this protocol was committed; its output is not kept.

## Decision rule

The diagnosis counts as **confirmed** only if all three hold in the even/world cell:

1. all four conditions of the damping/ decision rule pass on its 24 bodies, including the timers-shuffled control (paired seed-bootstrap 95% interval of control minus trained surprise left excludes zero);
2. its median surprise left in the second-event window is lower than the bunched/shared cell's, with the paired seed-bootstrap 95% interval of the difference excluding zero;
3. the local rule's anti-image score is at or above the timing-blind bound in at least 18 of its 24 bodies (bodies without exact bounds count as failures).

The even/shared and bunched/world cells are reported to attribute any change to one factor or the other, with no pass rule. Astra's positive reconstruction score is reported and is expected to stay negative.

If the even/world cell fails, the diagnosis is not confirmed, and the limit lies at least partly in the rule itself.
