# Frozen image-feature extension

Question: does the typed semantic interface work when its input is a frozen pretrained vision representation rather than our known linear factor encoding?

- Render object cards with three colors, three shapes and left/right positions. Two supplied card rows identify roles a and b. This assumes object separation; it does not test object detection.
- Use cached DINOv2 ViT-B/14 with four registers, pretrained weights loaded strictly, on CPU. Freeze every encoder parameter. Use 224×224 RGB images and ImageNet normalization.
- Exclude red circles at either position from compiler training and validation. All individual colors, shapes and positions remain represented. The pretrained encoder's original training set is outside this holdout claim.
- Render four training styles, two validation styles and two test styles, with distinct deterministic seeds for background, size and small location changes. Use the same style parameters across semantic states so style cannot identify a target.
- Extract all 18 object states in the eight styles, caching features and hashes. Training sees only 16 permitted states × four styles; validation sees only the same states in two new styles. Test includes all states in two further styles.
- Fit independent color, shape and position logistic readouts with fixed C=1, plus the explicit semantic-coordinate decoder with fixed ridge regression alpha=1 and nearest levels. Fit feature standardization using training examples only. Choose the reported primary readout using validation whole-object exactness, with logistic first on a tie.
- Evaluate all 34 held-out scene meanings and their test-style combinations as complete programs and executable queries. Report per-field accuracy, role swaps, same-style color counterfactuals, mismatched inputs, and familiar meanings in the test styles separately.
- Preserve failures; no parameter search or encoder fine-tuning is part of this extension.

These are rendered images, not real neural recordings or natural-image scenes. The added difficulty is a learned frozen encoder with an organization we did not design. Result scope remains the supplied object cards, roles and held-out compiler combinations.
