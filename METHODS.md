# Methods

## Origin and experimental scope

The original research goal was to translate neural-derived structure into explicit claims a receiving system could inspect and use. The decoder was developed within NISP, with partial full-pipeline integration. That work exposed two distinct questions: whether an input carries the intended information, and whether a decoder preserves its meaning in a symbolic expression.

This package makes the interface runnable on specified tasks. The proxy model is a new lightweight fit on recovered inputs, rather than a restored historical decoder checkpoint. The controlled-composition and DINOv2 experiments were added during recovery to examine the richer scene language and its executable semantics. Their targets and evidence are separate from the original neural-state interpretation.

## Model and language boundary

The front ends estimate discrete fields from continuous arrays. `SceneCompiler` uses three learned scalar coordinates with nearest semantic levels. `ImageFeatureCompiler` uses the validation-selected field readout for frozen visual features. `ProxyCompiler` predicts one of four proxy expressions.

The scene compiler checks each field and constructs one object for each supplied role, a and b. The language has 18 possible object descriptions and $18^2=324$ scene descriptions. Its typed round trip is exhaustively tested. Selection and count queries operate on the predicted properties; `left_of` compares predicted positions.

The numerical model is uncertain. The compiler and interpreter provide explicit meaning for its output. Accuracy is measured on complete programs, rather than inferred from syntax validity. The proxy path validates and renders its four-expression schema. The selection/count/order interpreter applies to the separate scene language; the proxy command performs no robot actuation or Lean proof discharge.

## Recovered proxy task

The 340 arrays are five augmentations of 68 original inputs: 53 cached TRIBE-derived UCF clip vectors and 15 synthetic scripted-robot vectors. Saved extraction indices identify the UCF branch as `mock:false`; extraction was not rerun during recovery. All 15 original robot vectors exactly match the legacy deterministic `_mock_latent` generator in NISP's `scripts/precompute_latents.py`. Older descriptions calling the robot vectors real TRIBE outputs conflict with that array-level comparison.

The four targets are scene normal/anomaly and robot proceed/abort; the historical conversion maps both REPLAN and ABORT to the latter label. The `(17,16)` input is a reshape/pooling adapter over feature coordinates, not a demonstrated anatomical-by-time organization. This is a mixed-source proxy benchmark, rather than 68 human neural observations.

The fitted model normalizes each array, appends a one-hot domain hint and applies a standardized linear readout:

$$
u=\mathrm{vec}\!\left(\frac{X-\overline X}{\mathrm{std}(X)+10^{-8}}\right)\mathbin{\oplus}e_d,
\qquad
\widehat{k}=\arg\max_k\left[W\frac{u-\mu}{s}+b\right]_k,
\qquad P=\mathcal{E}_{\widehat{k}}.
$$

Here concatenation adds the supplied scene/robot hint to the normalized features, and the feature scaler and class weights come from training. The output is selected from four fixed expressions. The hint does not mask classes: the recorded out-of-fold predictions include one robot-to-scene error. Keeping this behavior preserves the evaluated model; enforcing domain agreement would be an additional policy.

Four stratified original-input folds use seed 20261022. A stratified validation partition is taken inside each training fold. Augmented versions of an original never cross partitions. Feature scaling is fitted on training rows.

Fixed logistic regression scores 51/68 out of fold; fixed RBF SVM scores 50/68. The saved linear predictions comprise 37/53 video inputs and 14/15 synthetic robot inputs correct within those same mixed-source folds. This breakdown is not a separately trained video-only experiment. The repaired sequence model scores 42/68. The first sequence pilot scored 15/17 and was favorable relative to this broader evaluation. The final lightweight proxy model uses all available originals; its in-sample outputs are separate from the out-of-fold evidence.

```bash
python experiments/proxy.py
```

This command refits the fixed logistic and RBF baselines from the bundled arrays. It does not rerun sequence-model training. The sequence score is retained in the evidence record from the separate recovery evaluation.

Robot cycles come from one scenario, so cycle holdout does not establish transfer to new scenarios. The task labels measure the four proxy outcomes, rather than an independently isolated human judgment state or incremental information beyond matched sensor baselines.

## Frozen image-feature experiment

The encoder is DINOv2 ViT-B/14 with four registers, frozen throughout. We load its weights strictly and encode 224×224 RGB cards with ImageNet mean/std normalization. Training the compiler never changes the encoder.

There are 18 object states: three colors × three shapes × two positions. Cards vary in background, size and small placement offsets. Each style uses the same nuisance parameters across states. Four styles are for training, two for validation, and two for testing.

Both red-circle states are excluded from training and validation. This leaves 64 training cards and 32 validation cards. Test features contain all 18 states in two further styles. The 34 held-out two-object meanings each have four combinations of test styles, giving 136 programs. Object separation and role ordering are supplied.

Feature scaling is fitted on training cards only. Two readouts are predefined:

- Independent color, shape and position logistic heads, C=1.
- Three scalar coordinates fitted by ridge regression, alpha=1, decoded to nearest semantic levels.

Both have perfect validation object accuracy. The predefined tie order chooses logistic regression. On unseen meanings, it scores 136/136, compared with 64/136 for the scalar grid. Both score perfectly on familiar meanings in test styles. The scalar result illustrates sensitivity to the numerical representation of categories; it does not establish that every semantic-grid method fails.

The experiment concerns combinations absent from compiler training. DINOv2 was pretrained separately, and these are rendered cards rather than natural scenes or neural recordings. The source image hashes, encoder identity and checkpoint hash are in [the provenance record](results/image_feature_provenance.json).

Reproduce the readout fit from bundled features:

```bash
python experiments/image_features/run_experiment.py --features benchmarks/image_features.npz
```

For full extraction, the experiment expects the official DINOv2 source and pretrained weights in the local Torch cache. `DINO_REPOSITORY` and `DINO_WEIGHTS` override those locations. The script renders the cards, loads the encoder on CPU, records feature provenance and fits the readouts. No download occurs inside the experiment.

## Controlled composition experiment

A known orthonormal map mixes object factors and independent nuisance coordinates into 32 features. This benchmark uses 162 scene meanings with the two objects on opposite sides; the language itself allows 324 descriptions. Training excludes every scene containing red circle, leaving 128 training meanings and 34 held-out meanings. Train, validation and test have independent noise realizations. Models use seeds 11, 29 and 47, with validation-only checkpoint selection.

A semantic product grid with 18 codes per object reaches 34/34 unseen meanings in every seed, including the tested noisy views and single-color counterfactuals. A five-level, four-dimensional unstructured bottleneck has 625 possible codes but reaches 0/34 in all seeds. Its matched continuous ablation scores 16/34, 0/34 and 0/34. Thus code capacity and quantization alone do not explain the difference.

```bash
python experiments/composition/experiment.py --kinds factor_grid --output factor_grid_run
python experiments/composition/experiment.py --kinds linear quantized --output initial_run
python experiments/composition/experiment.py --kinds continuous_bottleneck --output continuous_run
```

The benchmark also enumerates 81 pairs of role-swapped scenes with exactly equal mean-pooled input and different programs. A deterministic pooled-input decoder can be correct on at most one member of each pair. The corresponding 50% ceiling is an input-information constraint; preserving roles removes that ambiguity. Role swapping in the shared readouts is handled equivariantly by construction.

## Relation to existing work

[Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html)
(Koh et al., 2020) predict supervised concepts as an intermediate interface to
a downstream prediction. The scene front ends here also predict named
properties. A deterministic interpreter then answers queries over those
properties. This makes the representation consumed by the next component
explicit and gives each predicted field an executable meaning.

The [Neuro-Symbolic Concept Learner](https://arxiv.org/abs/1904.12584)
(Mao et al., 2019) connects object-based visual representations to executable
programs, including tests of new attribute combinations. It learns concepts
and language from question-answer supervision. Our scene tasks supply object
roles, property labels and the query language, isolating the feature-to-field
translation and its generalization. The two evaluation protocols and reported
accuracies answer different questions.

The released experiments provide three useful cases for that interface:

- **Intermediate representation:** on the same frozen image features and
  predefined readouts, the validation-selected categorical heads produce
  136/136 held-out programs, while the ordinal scalar grid produces 64/136.
  The choice of representation for the fields changes the resulting programs.
- **Input information:** the controlled task enumerates 81 role-swapped pairs
  with identical mean-pooled features and different meanings. This establishes
  the 50% ceiling for a deterministic decoder receiving that pooled input.
  The supplied-role implementation preserves the information lost by pooling.
- **Grounding and validity:** exhaustive round trips cover the 324 meanings
  accepted by the scene language. Held-out program accuracy separately tests
  whether the readout assigns the intended meaning to its input.

The contribution is a portable implementation and controlled comparison of
these design choices, alongside the recovered four-target proxy task. The
broader research question remains faithful translation from neural-derived
features into useful symbolic structure.

## Evidence and runtime

[Evidence](results/evidence.json) records counts, per-original proxy predictions and source-result hashes. [Asset hashes](results/asset_hashes.json) identify the shipped parameters and examples. The wheel includes only lightweight parameters and feature examples; the separate 346 MB DINOv2 checkpoint is unnecessary for normal compiler inference.

The proxy runtime preserves the float32 normalization used during fitting. Inputs must have the correct shape and finite values, and remain finite through conversion, mean/standard-deviation computation and normalization. Unsupported magnitudes raise `ValueError` before classification. Constant arrays retain the previous defined behavior; this check is numerical validation rather than a learned confidence or abstention rule.

The test suite covers all 324 scene-language round trips, saved image-task results, bundled CLI examples, malformed inputs and proxy normalization overflow. A regression comparison checks unchanged proxy predictions against the previous inference arithmetic on all 340 bundled rows. These checks qualify software behavior; the held-out experiments provide the task-accuracy evidence.
