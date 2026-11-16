# Predictive damping 2 — results

16 November 2026. Built and run by Claude Opus 5.5 as a follow-up to [damping/](../damping/README.md). [Protocol](PROTOCOL_DAMPING2.md), [runner](run_damping2.py) and [analysis](analyze_damping2.py) were committed before the batch ran; the receipt records hashes of the protocol and runner, which match. One plumbing fix was made to the analysis after the run (cell names come back from JSON as lists; `tuple(c)` in one line). It changes no measurement. [Receipt](results_damping2.json) · [summary](summary_damping2.json).

**Result: the diagnosis is not confirmed.** Spreading the timers out did make the local rule use timing. But the second predicted event is still not cancelled in any version of the bank, so the bank was not why the second event failed. The rule is.

## What was tested

In damping/, the rule cancelled the first predicted event and barely touched the second, and the timers-shuffled control cost too little. A post-hoc analysis blamed the timer bank: two thirds of the timers fired in one ten-tick bunch, and the body's own anti-waves moved the slow ones. This batch fixed each of those, alone and together, on fresh seeds (140001–140012), with the rule, its settings, scoring and controls unchanged.

| Factor | Original | Changed |
|---|---|---|
| Spacing | bunched: 16 of 24 timers fire within ticks 51–61 | even: timers fire about 4 ticks apart over ticks 51–150 (70–85 and 119–140 cannot be reached by this timer) |
| Timer input | shared field: the body's own writes reach its clock | world only: timers hear the stimuli's wave, not the body's output |

24 bodies per cell (12 schedules × ABC and ACB), three events, jittered training gaps. Zero clipped frames in every cell.

## Results

Medians over 24 bodies. Surprise left is what remains of the missing wave when the learned sequence arrives (100% = untrained). The first-event window runs from the first missing event to the second; the second-event window from the second missing event to 100 frames later.

| Cell | Surprise left | First-event window | **Second-event window** | Anti-image (A only) | Own history | Timers-shuffled control, 95% interval | Rule ≥ timing-blind bound |
|---|---:|---:|---:|---:|---:|---|---:|
| bunched / shared (original design) | 55% | 12% | **98%** | 0.41 | 24/24 | +0.08 to +0.24 | 6 of 23 |
| even / shared | 62% | 23% | **114%** | 0.28 | 23/24 | +0.28 to +0.47 | 13 of 20 |
| bunched / world | 54% | 13% | **125%** | 0.38 | 24/24 | +0.02 to +0.25 | 8 of 24 |
| even / world | 57% | 25% | **119%** | 0.34 | 24/24 | +0.26 to +0.54 | 20 of 24 |

Exact linear bounds (see [damping/linear_bound.py](../damping/linear_bound.py)) for the trained bodies:

| Cell | Timing-blind bound | Ceiling | Largest weight the ceiling needs |
|---|---:|---:|---:|
| bunched / shared | 0.32 | 0.69 | 153 |
| even / shared | 0.15 | 0.66 | 1.9 |
| bunched / world | 0.40 | 0.98 | 29 |
| even / world | 0.19 | 0.96 | 1.5 |

Astra's original positive reconstruction score is negative in every cell (−1.32 to −1.46), as declared.

## Against the decision rule

| Condition (even / world cell) | Result |
|---|---|
| 1. All four damping/ conditions pass, including the timers-shuffled control | **pass** (timers-shuffled +0.26 to +0.54) |
| 2. Second-event window lower than bunched / shared, interval excluding zero | **fail**: it is *higher*, by 0.13 (interval −0.23 to −0.02 for bunched minus even) |
| 3. Rule at or above the timing-blind bound in at least 18 of 24 | **pass**: 20 of 24 |

Diagnosis **not confirmed**.

## What it means

1. **With the timers spread out, the rule does use timing.** Shuffling which timer drives which branches now costs a lot, and in 20 of 24 bodies the rule does better than any body that ignores timing could. That part of the diagnosis held.
2. **The second event is not predicted in any cell.** Its window ends up with as much energy as an untrained body or more (98–125%). The trained bodies cancel the first missing event well (12–25% left) and then add energy afterwards.
3. **The bank is not the limit any more; the rule is.** With even, world-fed timers, perfect weights would cancel 96% of the missing wave using weights no larger than 1.5. The rule reaches 0.34.
4. **The original near-fail did not replicate.** The unchanged design on fresh seeds passes all four damping/ conditions, including the timers-shuffled control (+0.08 to +0.24). Last night's narrow fail (−0.01 to +0.15) was at the edge of chance in both directions.

## A likely reason, not tested

The rule listens at its branch tip for steps 3–5 after its own pulse and nowhere else. The wave a ping makes rings for about a hundred frames. So each branch learns to flatten the first few frames of what follows its own firing and is blind to the ringing it leaves later. The first event's anti-waves keep ringing into the second event's window, and no branch is ever told about that. That would explain why the second-event window gets worse than passive in exactly the bodies that cancel the first event best.

If this is right, a rule that listens over a longer window, or that is driven by the residual at the time the second event is due, should help. That is a guess and would need its own pre-registered test.

## Limits

- Twelve schedules per cell, deterministic, one geometry, no noise.
- Bodies whose timers fired more than once on the cue have no exact bounds (1, 4, 0 and 0 bodies in the four cells); for the decision rule they counted as failures.
- The "world only" input is an architectural separation, not a claim about biology.
- Cancelling on schedule is learned timing, not inference of hidden causes.

## Reproduce

```sh
python damping2/run_damping2.py --workers 2      # about 8 minutes on 2 cores
python damping2/analyze_damping2.py
```
