# Review: a channel is not yet a learned explanation

15 November 2026. Source review and proposed experiments; no new learning result is claimed here.

The negative recall result is informative. Experiment 009's outside linear reader could identify a body's training history from its probe response. This assay asks the body itself to emit the missing sensory consequence, without such a reader. Recoverable history does not guarantee that particular behavior.

## What the implementation supports

`run.py` rebuilds fast state between episodes, carries only weights/alive flags, freezes learning for probes, and constructs both targets from passive simulations. Targets enter scoring only. The baseline equivalence check compares weights, surviving branches and firing times against the original 009 engine. The state payload is selected from the sender's history; the receiver has no specially trained shape decoder.

The packet's two samples use the same connection weights, and growth updates once per firing. Shape changes therefore affect wave interference and subsequent event timing. Equal packet area and squared-drive sum do not imply equal energy injected into a driven wave field, nor equal total energy over an episode. Any future efficiency claim needs measured drive work, as well as event counts.

The inverse control receives 52 scalar observations (two sensors × 26 frames), not just two numbers. Its three-column dictionary has full rank. With known locations, onset and medium, fitting three amplitudes is deliberately easy. Predicting withheld sensors is meaningful within this supplied model, but it does not show that a population discovered the model.

## The next experiment worth doing

First separate **learning the map** from **encoding a message**. A minimal next gate could learn a forward response dictionary from independent calibration episodes instead of supplying exact unit-source responses. Freeze it, infer amplitudes from partial observations, and predict withheld sensors on fresh mixtures. Compare the learned dictionary with the exact-physics upper bound, a simple direct linear predictor, zero prediction, and shuffled source/sensor addresses. This would establish learned system identification in a narrow setting; it would still not establish local neuronal learning.

Then test the proposed local mechanism against that attainable reference. Choose an explicit prediction target and local eligibility rule before the batch. Compare learning-on, learning-off, shuffled teaching error and an ordinary fitted linear baseline. Keep development seeds separate from final seeds. Score unseen sensor/time windows and report inference failures, clipping, firing counts and drive budgets. Only after the learning gate works should identical, fixed-shaped and state-shaped pings be compared under the same rule and budgets.

For the separate associative-recall question, the cue must actually identify the desired continuation through the learned structure or an independently available context. An A-only cue can select ABC versus ACB here because each body was trained on only one history. If both histories are stored in the same body, expecting A alone to select one introduces ambiguity; add a disambiguating cue or measure a predictive distribution. That is different from experiment 011's question of whether two histories remain readable after hardening.

Antiphase cancellation could become an error signal, but cancellation alone does not identify a hidden cause. Test it by moving or silencing a source and asking whether the inferred explanation predicts the resulting change at an unseen sensor. Successful suppression at the observed sensor alone is insufficient.

These are proposed discriminators, not measured successes. Preserve the original negative result when adding any later experiment.

## Predictive-damping follow-up

During this review Fable committed `damping/PROTOCOL_DAMPING.md`, explicitly testing Gemini's proposal. This changes both the rule and the body: 24 timers at one aperture, signed branches starting at zero, and a local update driven by the field at each branch tip after a firing. It uses fixed impulses, not state-shaped packets.

The protocol correctly keeps Astra's original positive reconstruction score while separately testing expected-input suppression and an omitted-input **negative** wave. Passing cancellation would be evidence for learned associative anticipation in this setup, not recovery of hidden causes or success on the original positive-wave criterion. The sign-flipped rule, destination shuffle and timer shuffle address useful alternatives. Fresh schedule seeds follow development tuning, which is disclosed.

A further discriminator is timing matched to an ordinary learned linear predictor: the larger bank supplies many delays absent from Astra's three-unit body. Improvements cannot be attributed to the learning rule alone while architecture, thresholds and initial weights also change. A fixed/passive baseline and the reported positive score should remain visible. No full damping result was available at this checkpoint; this review has not run that new batch.
