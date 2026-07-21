import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Spatial3DFourierEncoding(nn.Module):
    """
    3D Sinusoidal Positional Encoding for MNI coordinate grids.

    Overcomes the spectral bias of standard attention networks by mapping
    discrete (x, y, z) MNI indices into a high-frequency continuous Fourier
    feature space. Each spatial axis receives d_model // 3 dedicated dimensions,
    split evenly between sin and cos at logarithmically spaced frequencies.

    REQUIREMENT: d_model must be divisible by 6 (2 per sin/cos × 3 axes).

    The base denominator (default 10000.0) should be tuned to the spatial
    coordinate range. For a 96×96×96 MNI grid (values 0–95), run the
    Bounding Volume Analysis in §5 to verify frequency coverage before
    committing to this value.

    coords : (B, P, 3)  int64 — MNI grid indices, values in [0, 95]
    returns: (B, P, d_model) — spatially grounded feature embeddings
    """
    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        if d_model % 6 != 0:
            raise ValueError(
                f"d_model must be divisible by 6 for 3D Fourier encoding. "
                f"Got d_model={d_model}. Use 768 (divisible by 6) or adjust."
            )
        self.d_model   = d_model
        self.dim_chunk = d_model // 3   # dimensions allocated per spatial axis
        self.base      = base

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        B, P, _ = coords.shape
        pe = torch.zeros(B, P, self.d_model, device=coords.device,
                         dtype=torch.float32)

        x = coords[:, :, 0].float()   # (B, P)
        y = coords[:, :, 1].float()
        z = coords[:, :, 2].float()

        # Logarithmically spaced frequency decay per axis
        div_term = torch.exp(
            torch.arange(0, self.dim_chunk, 2,
                         dtype=torch.float32, device=coords.device)
            * -(math.log(self.base) / self.dim_chunk)
        )   # (dim_chunk // 2,)

        c = self.dim_chunk

        # X-axis: [0 : c]
        pe[:, :, 0:c:2] = torch.sin(x.unsqueeze(-1) * div_term)
        pe[:, :, 1:c:2] = torch.cos(x.unsqueeze(-1) * div_term)

        # Y-axis: [c : 2c]
        pe[:, :, c:2*c:2]   = torch.sin(y.unsqueeze(-1) * div_term)
        pe[:, :, c+1:2*c:2] = torch.cos(y.unsqueeze(-1) * div_term)

        # Z-axis: [2c : 3c]
        pe[:, :, 2*c::2]   = torch.sin(z.unsqueeze(-1) * div_term)
        pe[:, :, 2*c+1::2] = torch.cos(z.unsqueeze(-1) * div_term)

        return pe


class PerceiverBottleneck(nn.Module):
    """
    Perceiver IO latent bottleneck.

    Maps (B, P, D) patch tokens → (B, K, D) latent slots.

    Complexity:
      Cross-attention: O(K × P)   vs O(P²) for full self-attention
      Self-attention:  O(K²)      negligible for K ≤ 128

    K is data-dependent — run §5.1 and §5.4 before committing.
    Use K=64 as default; upgrade to K=128 if spatial sparsity < 40%.

    anatomical_init : (K, 3) optional — MNI centroids for warm-start.
    """
    def __init__(self,
                 d_model:     int   = 768,
                 num_latents: int   = 128,
                 num_heads:   int   = 8,
                 num_layers:  int   = 6,
                 dropout:     float = 0.1):
        super().__init__()
        self.K = num_latents

        # Learnable latent query bank — initialized with scaled noise
        # Call set_anatomical_queries() after construction for warm-start
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, d_model) * (d_model ** -0.5)
        )

        # Cross-attention: K queries attend to P patch tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            batch_first=True, dropout=dropout
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # Deep self-attention within the compressed K-token latent space
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,   # Pre-LN: more stable for fMRI
            activation='gelu', dropout=dropout
        )
        self.latent_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # CLS token fusion — gated blend of Omni-fMRI global summary
        # The CLS token from sceneX_cls.npy captures holistic temporal context
        # not present in any individual patch. Gating prevents CLS from
        # overriding spatial diversity when it carries low-information windows.
        self.cls_proj = nn.Linear(d_model, d_model)
        self.cls_gate = nn.Linear(d_model, d_model)

    def set_anatomical_queries(self, mni_centroids: torch.Tensor,
                                coord_encoder: Spatial3DFourierEncoding):
        """
        Warm-start the K latent queries from K anatomical MNI centroids.
        Recommended: use KMeans(n_clusters=K) on coords.npy as centroids.
        See §5.4 for the extraction protocol.

        mni_centroids : (K, 3) int64 — MNI grid coordinates
        coord_encoder : Spatial3DFourierEncoding — use the model's own pe_3d
        """
        with torch.no_grad():
            pe = coord_encoder(
                mni_centroids.unsqueeze(0).float()   # (1, K, 3)
            ).squeeze(0)                              # (K, d_model)
            self.latent_queries.data[0].copy_(pe * 0.1)

    def forward(self,
                x:         torch.Tensor,   # (B, P, D) — coord-enriched patch tokens
                pad_mask:  torch.Tensor,   # (B, P) bool — True=valid
                cls_token: torch.Tensor    # (B, D) — from sceneX_cls.npy
                ) -> torch.Tensor:
        B = x.size(0)

        # Broadcast learned queries over batch
        latents = self.latent_queries.expand(B, -1, -1)   # (B, K, D)

        # Cross-attend: K latents query P patch tokens
        # key_padding_mask: True = IGNORE (PyTorch convention — inverted from data mask)
        attended, _ = self.cross_attn(
            query=latents,
            key=x,
            value=x,
            key_padding_mask=~pad_mask    # ~mask: True = dead padding, ignore
        )
        latents = self.cross_norm(latents + attended)    # (B, K, D)

        # Deep self-attention within the K-token latent space
        latents = self.latent_transformer(latents)        # (B, K, D)

        # Gated CLS fusion: global brain state as conditioning signal
        cls_proj = self.cls_proj(cls_token)               # (B, D)
        gate     = torch.sigmoid(self.cls_gate(cls_proj)) # (B, D) ∈ (0, 1)
        latents  = latents * gate.unsqueeze(1)            # (B, K, D)

        return latents


class DeltaEncoder(nn.Module):
    """
    Delta encoding stack: patch tokens → topographic K-slot latents.

    Forward pass:
      1. Inject 3D MNI coordinate encodings into patch embeddings
         (overcomes spectral bias; adds spatial topography to permutation-
         invariant tokens)
      2. Compress P=4345 tokens → K latent slots via Perceiver IO
         (O(K×P) cross-attention instead of O(P²))
      3. Fuse global CLS token via gated blend
    """
    def __init__(self, d_model: int = 768, K: int = 128,
                 fourier_base: float = 10000.0):
        super().__init__()
        self.pe_3d      = Spatial3DFourierEncoding(d_model, base=fourier_base)
        self.bottleneck = PerceiverBottleneck(d_model, num_latents=K)

    def forward(self,
                patch_tokens: torch.Tensor,   # (B, P, 768)
                coords:       torch.Tensor,   # (B, P, 3)
                mask:         torch.Tensor,   # (B, P) bool
                cls_token:    torch.Tensor    # (B, 768)
                ) -> torch.Tensor:

        # Step 1: Enrich patch embeddings with 3D spatial position
        # [ABLATION 1]: Removed spatial encoding. The model is now completely blind to 3D topography.
        x = patch_tokens # No PE injection    # (B, P, 768)

        # Step 2 + 3: Perceiver bottleneck + CLS gated fusion
        z_brain_spatial = self.bottleneck(x, mask, cls_token)  # (B, K, 768)

        return z_brain_spatial
