# Neural-to-eDSL Compiler

Earlier neuro-symbolic formalization project. The system takes neural latent structure, maps it into an embedded DSL, and targets Lean 4-style formal objects. The point was whether a noisy internal state could be made strict enough for symbolic checking.

## What It Shows

- A neural-to-symbolic architecture with finite/discrete bottlenecks, grammar-aware generation, and formal-output pressure.
- A set of parser, grammar, network, and training components that connect latent representations to structured eDSL outputs.
- A research trail showing the cost of trying to force neural structure directly into formal objects.

## What It Does Not Claim

- It does not establish the current human measurement claim elsewhere in this application.
- It does not prove that a human phenomenon was isolated.

## Files

- `models/`: parser, grammar, network, autoencoder, and delta components.
- `training/`: training loops and ablations.
- `metrics/`: loss and metric helpers.
- `dataloader/`: split and preprocessing utilities.
- `evidence/`: selected research/engineering logs retained as provenance.

## Reproducibility Note

This prepared bundle intentionally removes large checkpoints, raw latent arrays, external run archives, deploy scripts, and local secrets. It is an evidence snapshot, not a one-command reproduction archive.
