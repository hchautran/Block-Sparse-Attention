"""
Profile each sub-component of the SAM2/SAM3 Hiera image encoder.

Architecture (SAM2-Hiera-Large, 1024×1024 input):
  PatchEmbed            Conv2d 7×7 / stride 4   →  (B, 256, 256, 144)
  Stage 1  [2 blocks]   window_size=8,  heads=2  →  (B, 128, 128, 288)
  Stage 2  [6 blocks]   window_size=4,  heads=4  →  (B,  64,  64, 576)
  Stage 3  [36 blocks]  window_size=14, heads=8  →  (B,  32,  32,1152)
    ↳ last `global_att_blocks` use global attn instead of window attn
  Stage 4  [4 blocks]   global attn,   heads=16  →  (B,  16,  16, …)
  Neck (FPN)            multi-scale Conv + LayerNorm → 4× (B, 256, H, W)

Profiled granularity:
  • Per-stage total time
  • Per-block time (aggregated: mean/min/max across blocks in a stage)
  • Intra-block breakdown: LayerNorm | Attention | MLP

Usage:
    # Mock mode (no SAM2 install needed):
    python profile_sam3_image_encoder.py --mock

    # Real SAM2:
    python profile_sam3_image_encoder.py \\
        --checkpoint /path/to/sam2_hiera_large.pt \\
        --model_cfg   sam2_hiera_l.yaml

    # Choose model size:
    python profile_sam3_image_encoder.py --mock --model_size large
    python profile_sam3_image_encoder.py --mock --model_size base_plus
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Model configs  (matches SAM2 YAML configs)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "tiny": dict(
        embed_dim=96, num_heads=(1, 2, 4, 8),
        stages=(1, 2, 7, 2), window_spec=(8, 4, 14, 7),
        global_att_blocks=(5, 7, 9), # block indices in stage 3 that use global attn
        neck_out_dim=256,
    ),
    "small": dict(
        embed_dim=96, num_heads=(1, 2, 4, 8),
        stages=(1, 2, 11, 2), window_spec=(8, 4, 14, 7),
        global_att_blocks=(7, 10, 13),
        neck_out_dim=256,
    ),
    "base_plus": dict(
        embed_dim=112, num_heads=(2, 4, 8, 16),
        stages=(2, 3, 16, 3), window_spec=(8, 4, 14, 7),
        global_att_blocks=(12, 16, 20),
        neck_out_dim=256,
    ),
    "large": dict(
        embed_dim=144, num_heads=(2, 4, 8, 16),
        stages=(2, 6, 36, 4), window_spec=(8, 4, 14, 7),
        global_att_blocks=(23, 33, 43),
        neck_out_dim=256,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Timing utilities
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stats:
    latencies_ms: List[float] = field(default_factory=list)
    peak_mem_mb: float = 0.0

    @property
    def mean(self): return sum(self.latencies_ms) / max(len(self.latencies_ms), 1)
    @property
    def mn(self):   return min(self.latencies_ms) if self.latencies_ms else 0.0
    @property
    def mx(self):   return max(self.latencies_ms) if self.latencies_ms else 0.0


@contextmanager
def cuda_timer(stats: Stats, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize(device)
    stats.latencies_ms.append(start.elapsed_time(end))
    peak = torch.cuda.max_memory_allocated(device)
    stats.peak_mem_mb = max(stats.peak_mem_mb, max(0, peak - mem_before) / 1024**2)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks  (faithful to SAM2 Hiera)
# ─────────────────────────────────────────────────────────────────────────────

def window_partition(x: torch.Tensor, window_size: int):
    """(B, H, W, C) → (num_windows*B, ws, ws, C), pad_hw"""
    B, H, W, C = x.shape
    # Pad so H and W are multiples of window_size (matches SAM2 behaviour)
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))   # pad last two spatial dims
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C), Hp, Wp


def window_unpartition(x: torch.Tensor, window_size: int, H: int, W: int, Hp: int, Wp: int):
    """(num_windows*B, ws, ws, C) → (B, H, W, C), crops padding"""
    B_times = x.shape[0]
    B = B_times // ((Hp // window_size) * (Wp // window_size))
    x = x.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous()


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, dtype=torch.float16):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1  = nn.Linear(dim, hidden, dtype=dtype)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, dim, dtype=dtype)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowedAttention(nn.Module):
    """Multi-head self-attention with optional window partitioning."""
    def __init__(self, dim: int, num_heads: int, window_size: int = 0, dtype=torch.float16):
        super().__init__()
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.scale       = self.head_dim ** -0.5
        self.window_size = window_size
        self.qkv  = nn.Linear(dim, dim * 3, bias=True,  dtype=dtype)
        self.proj = nn.Linear(dim, dim,     bias=True,  dtype=dtype)
        # Stores the last attention map shape for inspection: (batch_or_windows, heads, seq, seq)
        self.last_attn_shape: Optional[tuple] = None

    def forward(self, x: torch.Tensor):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        use_window = self.window_size > 0
        Hp = Wp = 0

        if use_window:
            x, Hp, Wp = window_partition(x, self.window_size)   # (nW*B, ws, ws, C)

        bx, h, w, c = x.shape
        x_flat = x.reshape(bx, h * w, c)

        qkv = self.qkv(x_flat).reshape(bx, h * w, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                          # (bx, heads, seq, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.last_attn_shape = tuple(attn.shape)          # (bx, heads, seq, seq)
        out  = (attn @ v).transpose(1, 2).reshape(bx, h * w, c)
        out  = self.proj(out).reshape(bx, h, w, c)

        if use_window:
            out = window_unpartition(out, self.window_size, H, W, Hp, Wp)
        return out


class HieraBlock(nn.Module):
    """
    One Hiera transformer block.
    Optionally includes a downsampling projection (at stage boundaries).
    """
    def __init__(self, dim: int, num_heads: int, window_size: int = 0,
                 downsample: bool = False, out_dim: Optional[int] = None,
                 mlp_ratio: float = 4.0, dtype=torch.float16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, dtype=dtype)
        self.attn  = WindowedAttention(dim, num_heads, window_size, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype)
        self.mlp   = MLP(dim, mlp_ratio, dtype=dtype)

        self.downsample = None
        if downsample:
            assert out_dim is not None
            self.downsample = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2, dtype=dtype)

    def forward(self, x: torch.Tensor):
        # x: (B, H, W, C)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        if self.downsample is not None:
            x = self.downsample(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x

    # ── fine-grained timing ──────────────────────────────────────────────────
    def forward_profiled(self, x: torch.Tensor, stats: Dict[str, Stats], device):
        with cuda_timer(stats["norm1"], device):
            n1 = self.norm1(x)
        with cuda_timer(stats["attn"], device):
            x = x + self.attn(n1)
        with cuda_timer(stats["norm2"], device):
            n2 = self.norm2(x)
        with cuda_timer(stats["mlp"], device):
            x = x + self.mlp(n2)
        if self.downsample is not None:
            ds_stats = stats.setdefault("downsample", Stats())
            with cuda_timer(ds_stats, device):
                x = self.downsample(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, kernel_size=7, stride=4, dtype=torch.float16):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size, stride=stride,
                              padding=padding, dtype=dtype)
        self.norm = nn.LayerNorm(embed_dim, dtype=dtype)

    def forward(self, x):
        x = self.proj(x)                          # (B, C, H', W')
        x = x.permute(0, 2, 3, 1)                # (B, H', W', C)
        return self.norm(x)


class FPNNeck(nn.Module):
    """
    SAM2 neck: 4-scale FPN producing uniform 256-dim feature maps.
    Input scales come from stage outputs.
    """
    def __init__(self, scale_dims: Tuple[int, ...], out_dim: int = 256, dtype=torch.float16):
        super().__init__()
        # lateral projections from each scale to out_dim
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, out_dim, 1, bias=False, dtype=dtype),
                nn.LayerNorm([out_dim, 1, 1]),   # placeholder shape (norm applied after)
            )
            for d in scale_dims
        ])
        self.convs = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False, dtype=dtype)
            for _ in scale_dims
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # features: list of (B, H, W, C) tensors, coarsest last
        outs = []
        prev = None
        for i, feat in enumerate(reversed(features)):
            feat = feat.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)
            lat  = self.laterals[i][0](feat)                # project channels
            lat  = F.layer_norm(lat, [lat.shape[1], lat.shape[2], lat.shape[3]])
            if prev is not None:
                lat = lat + F.interpolate(prev, size=lat.shape[-2:], mode="nearest")
            out  = self.convs[i](lat)
            outs.append(out)
            prev = out
        return list(reversed(outs))


# ─────────────────────────────────────────────────────────────────────────────
# Full Hiera Image Encoder (mock, matches SAM2 architecture)
# ─────────────────────────────────────────────────────────────────────────────

class HieraImageEncoder(nn.Module):
    def __init__(self, cfg: dict, dtype=torch.float16):
        super().__init__()
        base    = cfg["embed_dim"]
        heads   = cfg["num_heads"]
        stages  = cfg["stages"]
        windows = cfg["window_spec"]
        global_att_blocks = set(cfg.get("global_att_blocks", []))
        neck_out = cfg["neck_out_dim"]

        self.patch_embed = PatchEmbed(embed_dim=base, dtype=dtype)

        # Dimension of blocks within each stage (all blocks in a stage share the same dim).
        # Hiera doubles channels at every stage boundary via the downsampling conv.
        #   stage 0 blocks: base        → downsamples to base*2
        #   stage 1 blocks: base*2      → downsamples to base*4
        #   stage 2 blocks: base*4      → downsamples to base*8
        #   stage 3 blocks: base*8      → no downsample
        stage_block_dims = [base * (2 ** s) for s in range(4)]      # [C, 2C, 4C, 8C]
        stage_out_dims   = [base * (2 ** (s + 1)) for s in range(3)] + [base * 8]
        # stage_out_dims: what the last block of each stage outputs  [2C, 4C, 8C, 8C]

        self.stage_block_dims = stage_block_dims
        self.stage_out_dims   = stage_out_dims

        # Build blocks stage by stage
        self.stage_blocks: nn.ModuleList = nn.ModuleList()

        global_block_counter = 0  # counts blocks across all stages for global_att_blocks lookup
        for s_idx, (n_blocks, ws, h) in enumerate(zip(stages, windows, heads)):
            block_dim = stage_block_dims[s_idx]   # ALL blocks in this stage use this dim
            next_dim  = stage_out_dims[s_idx]      # output dim after downsampling
            stage     = nn.ModuleList()
            for b_idx in range(n_blocks):
                is_last   = (b_idx == n_blocks - 1)
                use_down  = is_last and (s_idx < 3)   # spatial+channel downsample at stage boundary
                is_global = global_block_counter in global_att_blocks
                eff_ws    = 0 if (s_idx == 3 or is_global) else ws   # 0 → global attn
                stage.append(HieraBlock(
                    dim=block_dim, num_heads=h,
                    window_size=eff_ws,
                    downsample=use_down,
                    out_dim=next_dim if use_down else None,
                    dtype=dtype,
                ))
                global_block_counter += 1
            self.stage_blocks.append(stage)

        # FPN neck iterates features coarsest-first (reversed), so pass dims in that order
        self.neck = FPNNeck(scale_dims=tuple(reversed(stage_out_dims)), out_dim=neck_out, dtype=dtype)
        self.stage_dims = stage_out_dims

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        stage_features = []
        for stage in self.stage_blocks:
            for blk in stage:
                x = blk(x)
            stage_features.append(x)
        return self.neck(stage_features)


# ─────────────────────────────────────────────────────────────────────────────
# Profiling logic
# ─────────────────────────────────────────────────────────────────────────────

def _sync_timer(fn, device):
    """Run fn() and return elapsed ms using CUDA events."""
    torch.cuda.synchronize(device)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize(device)
    return s.elapsed_time(e)


def _peak_mem(fn, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    m0 = torch.cuda.memory_allocated(device)
    fn()
    torch.cuda.synchronize(device)
    return max(0, torch.cuda.max_memory_allocated(device) - m0) / 1024**2


@torch.no_grad()
def profile_image_encoder(
    model: HieraImageEncoder,
    device: torch.device,
    dtype: torch.dtype,
    img_size: int = 1024,
    repeats: int = 10,
):
    B = 1
    x_proto = torch.randn(B, 3, img_size, img_size, device=device, dtype=dtype)

    stage_names  = [f"Stage {i+1}" for i in range(4)]
    stage_stats  = [Stats() for _ in range(4)]
    patch_stats  = Stats()
    neck_stats   = Stats()

    # per-block intra-component stats: stage → [block_idx → {norm1, attn, norm2, mlp, downsample?}]
    intra_stats: List[List[Dict[str, Stats]]] = []
    for s_idx, stage in enumerate(model.stage_blocks):
        stage_intra = []
        for b_idx, blk in enumerate(stage):
            d = {"norm1": Stats(), "attn": Stats(), "norm2": Stats(), "mlp": Stats()}
            if blk.downsample is not None:
                d["downsample"] = Stats()
            stage_intra.append(d)
        intra_stats.append(stage_intra)

    # ── warm-up ──────────────────────────────────────────────────────────────
    for _ in range(3):
        model(x_proto)
    torch.cuda.synchronize(device)

    # ── timed repeats ─────────────────────────────────────────────────────────
    for _ in range(repeats):
        x = x_proto.clone()

        # 1. Patch Embed
        with cuda_timer(patch_stats, device):
            x = model.patch_embed(x)

        stage_features = []

        # 2. Stages
        for s_idx, stage in enumerate(model.stage_blocks):
            # Stage-level timer
            x_stage_in = x.clone()
            with cuda_timer(stage_stats[s_idx], device):
                for blk in stage:
                    x = blk(x)
            stage_features.append(x)

            # Intra-block timing (re-run stage, so we don't add latency to stage timer)
            x_intra = x_stage_in.clone()
            for b_idx, blk in enumerate(stage):
                istats = intra_stats[s_idx][b_idx]
                x_intra = blk.forward_profiled(x_intra, istats, device)

        # 3. FPN Neck
        with cuda_timer(neck_stats, device):
            model.neck(stage_features)

    # ── Print results ─────────────────────────────────────────────────────────
    total_mean = patch_stats.mean + sum(s.mean for s in stage_stats) + neck_stats.mean

    print(f"\n{'='*70}")
    print(f"SAM2 Hiera Image Encoder Profile  |  img={img_size}×{img_size}  |  repeats={repeats}")
    print(f"{'='*70}")

    hdr = f"{'Component':<28} {'Mean ms':>9} {'Min ms':>9} {'Max ms':>9} {'Mem MB':>9} {'Share%':>8}"
    print(hdr)
    print("─" * 76)

    def row(name, s: Stats, total):
        pct = 100 * s.mean / total if total else 0
        print(f"{name:<28} {s.mean:>9.2f} {s.mn:>9.2f} {s.mx:>9.2f} {s.peak_mem_mb:>9.1f} {pct:>7.1f}%")

    row("PatchEmbed", patch_stats, total_mean)
    print()

    for s_idx, (sname, ss) in enumerate(zip(stage_names, stage_stats)):
        n_blocks = len(model.stage_blocks[s_idx])
        row(f"{sname}  [{n_blocks} blocks]", ss, total_mean)

        # Print attention map shape for each block type in this stage
        for b_idx, blk in enumerate(model.stage_blocks[s_idx]):
            shape = blk.attn.last_attn_shape
            if shape is not None:
                bx, heads, seq_q, seq_k = shape
                attn_type = "global" if blk.attn.window_size == 0 else f"window(ws={blk.attn.window_size})"
                # Only print once per distinct shape in the stage (avoid repeating identical rows)
                if b_idx == 0 or blk.attn.last_attn_shape != model.stage_blocks[s_idx][b_idx - 1].attn.last_attn_shape:
                    print(f"  {'  attn_map shape':<25}  ({bx}, {heads}, {seq_q}, {seq_k})  [{attn_type}]")

        # Per-block aggregate (sum across all blocks = cost per forward pass)
        # Components present in every block
        for comp in ("norm1", "attn", "norm2", "mlp"):
            vals = [intra_stats[s_idx][b][comp].mean for b in range(n_blocks)]
            mns  = [intra_stats[s_idx][b][comp].mn   for b in range(n_blocks)]
            mxs  = [intra_stats[s_idx][b][comp].mx   for b in range(n_blocks)]
            agg_mean, agg_mn, agg_mx = sum(vals), sum(mns), sum(mxs)
            pct     = 100 * agg_mean / total_mean if total_mean else 0
            blk_avg = agg_mean / n_blocks if n_blocks else 0
            print(f"  {'└─ ' + comp:<25} {agg_mean:>9.2f} {agg_mn:>9.2f} {agg_mx:>9.2f} {'':>9} {pct:>7.1f}%  (avg/blk={blk_avg:.3f}ms)")
        # Downsample exists only in the last block of stages 0-2
        ds_vals = [intra_stats[s_idx][b].get("downsample") for b in range(n_blocks)]
        ds_vals = [s for s in ds_vals if s is not None]
        if ds_vals:
            ds_mean = sum(s.mean for s in ds_vals)
            ds_mn   = sum(s.mn   for s in ds_vals)
            ds_mx   = sum(s.mx   for s in ds_vals)
            pct     = 100 * ds_mean / total_mean if total_mean else 0
            print(f"  {'└─ downsample (conv)':<25} {ds_mean:>9.2f} {ds_mn:>9.2f} {ds_mx:>9.2f} {'':>9} {pct:>7.1f}%")
        print()

    row("FPN Neck", neck_stats, total_mean)
    print("─" * 76)
    print(f"{'TOTAL':<28} {total_mean:>9.2f}")
    print(f"\nEstimated throughput: {1000/total_mean:.2f} fps")

    # ── Window vs Global attention breakdown ──────────────────────────────────
    print(f"\n{'─'*50}")
    print("Attention type breakdown per stage:")
    print(f"{'─'*50}")
    for s_idx, stage in enumerate(model.stage_blocks):
        win_ms, glob_ms = 0.0, 0.0
        win_n, glob_n   = 0, 0
        for b_idx, blk in enumerate(stage):
            attn_ms = intra_stats[s_idx][b_idx]["attn"].mean
            is_glob = (blk.attn.window_size == 0)
            if is_glob:
                glob_ms += attn_ms; glob_n += 1
            else:
                win_ms  += attn_ms; win_n  += 1
        parts = []
        if win_n:  parts.append(f"window({win_n} blks)={win_ms:.2f}ms")
        if glob_n: parts.append(f"global({glob_n} blks)={glob_ms:.2f}ms")
        print(f"  Stage {s_idx+1}: {' | '.join(parts)}")


# ─────────────────────────────────────────────────────────────────────────────
# Real SAM2 path
# ─────────────────────────────────────────────────────────────────────────────

model_cfg_used: dict = {}   # filled in main()


def extract_sam2_image_encoder(checkpoint: str, model_cfg: str, device: torch.device):
    """
    Load the real SAM2 image encoder.
    Works if `sam2` package is installed.
    """
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    sam2 = predictor.model
    sam2.eval()
    return sam2.image_encoder


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--model_cfg",   default=None)
    p.add_argument("--mock",        action="store_true",
                   help="Use synthetic mock model (no SAM2 install needed)")
    p.add_argument("--model_size",  default="large",
                   choices=list(MODEL_CONFIGS.keys()),
                   help="Mock model size (ignored when loading a real checkpoint)")
    p.add_argument("--img_size",    type=int, default=1024)
    p.add_argument("--repeats",     type=int, default=10)
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--dtype",       default="float16",
                   choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def main():
    global model_cfg_used

    args  = parse_args()
    dev   = torch.device(args.device)
    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32":  torch.float32}[args.dtype]

    if args.mock or args.checkpoint is None:
        cfg = MODEL_CONFIGS[args.model_size]
        model_cfg_used = cfg
        print(f"[mock] Building SAM2-Hiera-{args.model_size}  dtype={dtype}")
        encoder = HieraImageEncoder(cfg, dtype=dtype).to(dev).eval()
    else:
        print(f"Loading SAM2 from {args.checkpoint}")
        encoder = extract_sam2_image_encoder(args.checkpoint, args.model_cfg, dev)
        # approximate cfg for breakdown labels
        model_cfg_used = MODEL_CONFIGS.get("large", MODEL_CONFIGS["large"])

    with torch.no_grad():
        profile_image_encoder(encoder, dev, dtype, args.img_size, args.repeats)


if __name__ == "__main__":
    main()
