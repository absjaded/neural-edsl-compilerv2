# Neural-to-eDSL Compiler

Neuro-symbolic formalization project. The system takes neural latent structure, maps it into an embedded DSL, and targets Lean 4-style formal objects. The point was whether a noisy internal state could be made strict enough for symbolic checking.

## Core Question

Can a continuous internal representation be forced through enough structure that a symbolic system can inspect it?

This project treated that as a compiler problem: latent state in, discrete bottleneck, grammar pressure, formal object out. It was not a product UI. It was a research scaffold for making representation stricter than the substrate it came from.

## What Survived

- A neural-to-symbolic architecture with finite/discrete bottlenecks, grammar-aware generation, and formal-output pressure.
- A set of parser, grammar, network, and training components that connect latent representations to structured eDSL outputs.
- A research trail showing the cost of trying to force neural structure directly into formal objects.

## Repo Map

- `models/`: parser, grammar, network, autoencoder, and delta components.
- `training/`: training loops and ablations.
- `metrics/`: loss and metric helpers.
- `dataloader/`: split and preprocessing utilities.
- `evidence/`: selected research/engineering logs retained as provenance.
