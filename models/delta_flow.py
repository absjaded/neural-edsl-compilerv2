import torch
import torch.nn as nn

class SiTBlock(nn.Module):
    """
    Single DiT-style SiT transformer block with internal cross-attention
    conditioning on (B, K, D) Perceiver brain latents.
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Self-attention on trajectory x_t
        self.sa_norm = nn.LayerNorm(d_model)
        self.sa      = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        # Cross-attention: x_t queries Z_brain (B, K, D)
        self.ca_norm = nn.LayerNorm(d_model)
        self.ca      = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        # Feedforward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
        )

        # Adaptive LayerNorm: modulates scale/shift from timestep embedding
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model)   # 6 = shift+scale for sa, ca, ff
        )

    def forward(self, x: torch.Tensor,
                z_brain: torch.Tensor,
                t_emb:   torch.Tensor) -> torch.Tensor:
        """
        x       : (B, 1, D)  — noisy trajectory token
        z_brain : (B, K, D)  — Perceiver spatial latents (conditioning context)
        t_emb   : (B, D)     — timestep sinusoidal embedding
        """
        # Unpack adaLN modulation parameters
        shifts_scales = self.adaLN_modulation(t_emb)        # (B, 6D)
        s1, g1, s2, g2, s3, g3 = shifts_scales.chunk(6, dim=-1)  # each (B, D)

        def modulate(z, shift, scale):
            return z * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Self-attention with adaLN
        x_norm = modulate(self.sa_norm(x), s1, g1)
        x_sa, _ = self.sa(x_norm, x_norm, x_norm)
        x = x + x_sa

        # Cross-attention on Perceiver brain latents with adaLN
        x_norm = modulate(self.ca_norm(x), s2, g2)
        x_ca, ca_weights = self.ca(x_norm, z_brain, z_brain)
        x = x + x_ca

        # Feedforward with adaLN
        x_norm = modulate(self.ff_norm(x), s3, g3)
        x = x + self.ff(x_norm)

        return x, ca_weights   # ca_weights: (B, 1, K) — diagnostic


class SiTVelocityNet(nn.Module):
    """
    Scalable Interpolant Transformer velocity network.
    Predicts v_θ(x_t, t, Z_brain) — conditioned on the full K-slot brain latents.

    Input:  x_t    (B, D)  — noisy trajectory at time t
            t      (B,)    — flow time ∈ [0, 1]
            z_brain (B, K, D) — Perceiver spatial conditioning context
    Output: velocity (B, D) — predicted direction from x_t toward z_edsl
    """
    def __init__(self, d_model: int = 1664, n_heads: int = 8, n_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        # Project brain latents from Perceiver space (768) to SiT space (1664)
        self.brain_proj = nn.Linear(768, d_model)
        self.context_norm = nn.LayerNorm(d_model)

        # Timestep sinusoidal embedding
        self.t_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Trajectory token input projection
        self.x_in_proj  = nn.Linear(d_model, d_model)

        # DiT-style SiT blocks with cross-attention conditioning
        self.blocks = nn.ModuleList([
            SiTBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])

        # Output projection: velocity in eDSL space
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x_t: torch.Tensor,
                t:       torch.Tensor,
                z_brain: torch.Tensor) -> torch.Tensor:
        """
        x_t     : (B, D=1664)     — interpolated trajectory
        t       : (B,)             — flow time
        z_brain : (B, K, 768)     — Perceiver latents (conditioning)
        """
        B = x_t.shape[0]

        # Project brain latents to SiT dimensionality
        z_cond = self.brain_proj(z_brain)              # (B, K, 1664)
        c_pooled = self.context_norm(z_cond.mean(dim=1))

        # Timestep embedding
        t_emb = self.t_embed(t.unsqueeze(-1))          # (B, 1664)

        # Input trajectory as sequence token (B, 1, D)
        x = self.x_in_proj(x_t).unsqueeze(1)          # (B, 1, 1664)

        # Process through SiT blocks — each layer cross-attends to all K brain slots
        all_ca_weights = []
        for block in self.blocks:
            x, ca_w = block(x, z_cond, t_emb)
            all_ca_weights.append(ca_w)                # (B, 1, K) per layer

        # Output velocity
        v = self.out_proj(self.out_norm(x.squeeze(1))) # (B, 1664)
        return v, all_ca_weights   # ca_weights list: diagnostic for slot routing


    # ── ODE integration for inference ────────────────────────────────────────

    def flow_forward(self, z_brain: torch.Tensor, n_steps: int = 20,
                     z_start: torch.Tensor = None) -> torch.Tensor:
        """
        Integrate from noise x_0 ∼ N(0,I) → predicted z_edsl.
        Uses deterministic Euler method (or substitute Heun for 2nd-order).
        """
        B, d = z_brain.shape[0], self.out_proj.out_features
        x = z_start if z_start is not None else torch.randn(B, d, device=z_brain.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=x.device)
            v, _ = self.forward(x, t, z_brain)
            x = x + v * dt
        return x   # predicted z_edsl

    def flow_reverse(self, z_edsl: torch.Tensor, z_brain: torch.Tensor,
                     n_steps: int = 20) -> torch.Tensor:
        """Reverse ODE: z_edsl → z_brain_sem for cycle consistency."""
        x = z_edsl.clone()
        dt = 1.0 / n_steps
        for i in range(n_steps, 0, -1):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v, _ = self.forward(x, t, z_brain)
            x = x - v * dt
        return x
