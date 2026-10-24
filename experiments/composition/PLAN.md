# Composition and binding experiment

Question: can a learned continuous-input compiler produce correct executable descriptions for combinations absent from training, and preserve which object each property belongs to?

This is a new controlled construction. It is separate from the recovered TRIBE proxy data.

- Two named roles, `a` and `b`; each object has one of three colors, one of three shapes, and a left/right position. The two objects occupy opposite positions.
- A fixed random orthonormal mixing matrix maps categorical factors, continuous position and independent nuisance coordinates to 32-dimensional continuous object features. Input row identity supplies the role. The compiler receives no source labels at inference.
- There are 162 possible scene meanings. Exclude `red circle` from both objects in training and validation: 128 training meanings and 34 held-out meanings. Training and validation use separate noise realizations. Every single color and shape remains represented in training.
- Compare a shared linear factor readout with a small learned four-dimensional, five-level-per-dimension bottleneck and factor heads. Both compile their predicted fields through the same typed language. The linear alternative tests whether the representation preserves the needed information; the quantized alternative tests whether a compact learned bottleneck retains it.
- Use seeds 11, 29 and 47. Fixed training settings: AdamW 0.003, weight decay 0.0001, batch 128, maximum 150 epochs, validation every five epochs, 25-epoch patience. Checkpoint selection uses only noisy validation versions of training meanings. No selection on the held-out composition set.
- Report whole-program exactness, individual-field accuracy, syntax/type acceptance, executable query agreement, role-swap equivariance, and single-property counterfactual fidelity. Report clean held-out meanings and noisy variants separately.
- Use mismatched scenes as an input-dependence control. This is distinct from swapping roles: a correct compiler should change its role assignments when rows swap.
- Enumerate exact mean-pooling collisions. Each scene and its role-swapped counterpart have identical pooled input but different program meanings. Any deterministic pooled-input compiler can be exact on at most one member of each pair: an exact-match ceiling of 50% on the balanced paired task. Compute the pair counts, do not infer this ceiling from one failed training run.
- Test parser round trips, rejection of invalid fields/duplicate identities, interpreter behavior and the shared inference command.

The representation has deliberately simple, known factors. Success would establish execution and controlled compositional behavior of this compiler construction, without demonstrating neural-state extraction or generalization to real neural recordings.

## Follow-up after the initial comparison

All three shared linear readouts compiled the 34 held-out meanings correctly, whereas all three unstructured quantized bottlenecks had zero whole-program exactness on those meanings. Test a third model using one explicit scalar coordinate per semantic field: three color levels, three shape levels and two position levels, an 18-code product per object. Use a learned linear encoder, nearest semantic levels for discrete decoding, and distance-based categorical logits for the same supervised factor losses. Retain the same data, seeds, optimizer, validation selection and controls. This tests whether a structured discrete interface succeeds where the tested unstructured bottleneck failed; it is not a claim that all unstructured bottlenecks must fail.

Also test a matched continuous ablation of the unstructured quantized model: the same MLP, four bounded coordinates and output heads, without rounding. Keep all settings and seeds fixed. This distinguishes failure of the tested representation/architecture from failure specifically attributable to its quantization operation. Report the result even if it weakens the initial interpretation.
