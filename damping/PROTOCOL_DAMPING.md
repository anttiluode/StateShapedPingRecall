# Predictive damping: protocol

Written 15 November 2026 by Claude Fable 5.2, **before** the scientific batch. Follow-up to Astra's pilot in this repository (`PROTOCOL.md`, `REPORT.md`), testing Gemini's proposal.

## Question

Astra's pilot found that the existing growth rule, with or without state-shaped pings, makes an A-only cue produce a response *further* from the missing wave than doing nothing. Gemini proposed changing the rule, not the ping: let each branch carry a signed weight and adjust it to **flatten the field at its own tip** just after its unit fires. Does that rule make a body predict the rest of a learned sequence?

## What changes and what does not

Unchanged: the vendored upstream `WaveFieldNode` (96×96, speed 0.5, four substeps, damping 0.006), the vendored `FadingHistorySpikeNode`, host step order, stimulus sites A = 0, B = 7, C = 4, pulse sizes (stimulus 0.12 for two frames, ping 0.24), full wipe of fast state between episodes, Astra's schedule generator, Astra's probe (onset 23, amplitude 1, nominal gap) and Astra's scoring window.

Changed:

1. **Rule.** New `DampingReturnNode`: eight signed branches per unit, all starting at zero. After its unit fires, a branch listens to the field at its tip for node steps 3–5 and moves its weight by `-learning_rate * mean / self_response`. No budget, no pruning, no hardening. Local inputs only.
2. **Body.** A bank of 24 fade-triggered timers, decays 0.780, 0.788, …, 0.964, all listening at aperture 1 (next to A). Different fades fire at different delays after A. Arm threshold 0.80 and release 0.20 of each timer's own passive-A peak trace.
3. **Ping.** Identical one-frame pings only. State-shaped pings are not used here.

## Settings and how they were chosen

Chosen on development runs with seed 991 and hand-picked gaps, then locked: learning rate 0.03, listening window steps 3–5, `self_response` 0.71 (measured), 24 timers, arm threshold 0.80, 20 training episodes of 360 frames.

Two things were learned in development and are reported in the README as development observations, not results: with the learning rate at 0.3 the bank overshot (many timers correct the same error at once); with the arm threshold at 0.45 the timers were restarted by the later events' waves and three-event sequences failed.

## Batch

Twelve fresh schedules, seeds 130001–130012, generated exactly as in Astra's `run.py` (gap 28..42, amplitude 0.90..1.12, onset ±5 per episode).

- **Primary task: three events, jittered.** Each schedule trains one ABC body and one ACB body with the training gap varied ±3 per episode (Astra's schedule). 24 bodies.
- Secondary: three events, regular (no gap jitter). 24 bodies.
- Secondary: two events (A then B), jittered and regular. 12 bodies each.
- Control: the same rule with its sign flipped (amplifying), three events jittered. 24 bodies.

## Measurements

All with growth frozen, from a full wipe. `base` is the passive A-only wave; `T` is the passive full-sequence wave minus `base` (the missing wave); `W` is Astra's window, from two frames after the first missing event to 100 frames after the last.

- **Expected input.** Play the body its own sequence. *Surprise left* = energy of (response − base) in W divided by energy of T. 100% is a body that predicts nothing; 0% is perfect cancellation.
- **Omitted input.** Play A alone; `p` = response − base.
  - *Astra's reconstruction score*, `1 − |p − T|² / |T|²`, reported unchanged.
  - *Anti-image score*, `1 − |p + T|² / |T|²`: 1 means the body produced the exact negative of the missing wave.
  - *Own history preferred*: is `p` closer to the negative of its own missing wave than to the negative of the other order's?
- **Unexpected input.** Play the body the *other* order; surprise left, against that other sequence's missing wave.
- **Controls on each trained body:** destinations shuffled (each timer's eight weights permuted across sites); timers shuffled (weight rows permuted between timers, so the same destinations are driven by the wrong fades).
- **Which fade is chosen** (two-event bodies): the A-only firing time of the timer with the largest weight magnitude at B's site, against B's onset.

## Decision rule for the primary task

The damping rule counts as **producing sequence prediction** only if all hold over the 24 primary bodies:

1. median surprise left, expected input, at or below 75%;
2. median anti-image score, omitted input, at or above 0.10;
3. own history preferred in at least 18 of 24;
4. trained bodies beat both shuffled controls on surprise left, with the 95% bootstrap interval over the 12 schedules excluding zero (Astra's method).

**Declared in advance:** this rule produces the *negative* of a predicted wave. By Astra's original criterion (positive reconstruction score above 0.10) it is expected to fail, and that score will be reported as it comes out. Passing the rule above is a claim about prediction by cancellation, which is a different thing from reconstruction.

Also declared: cancelling a wave on schedule is associative timing, not inference of hidden causes. Astra's distinction in `REPORT.md` stands whatever happens here.
