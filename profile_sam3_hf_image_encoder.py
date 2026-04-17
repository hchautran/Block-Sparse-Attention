"""
Profile each sub-component of the SAM3 (HuggingFace) image encoder.

Architecture (facebook/sam3):
  Sam3VisionModel
  ├── backbone  (Sam3ViTModel)
  │   ├── embeddings  (patch_embed + position_embed)
  │   ├── layers[0..31]  (Sam3ViTLayer × 32)
  │   │   ├── layer_norm1
  │   │   ├── rotary_emb
  │   │   ├── attention   (Sam3ViTRoPEAttention)
  │   │   │   └── q/k/v/o_proj
  │   │   ├── layer_norm2
  │   │   └── mlp
  │   └── layer_norm  (final)
  └── neck  (Sam3VisionNeck — 4-scale FPN)
      └── fpn_layers[0..3]  (Sam3FPNLayer)

Window attention  → layers NOT in global_attn_indexes  (default: all except 7,15,23,31)
  attn_map shape  (B × n_windows, heads, ws², ws²)
  e.g. for 1008px: B×9, 16, 576, 576

Global attention  → layers at global_attn_indexes=[7,15,23,31]
  attn_map shape  (B, heads, H_tok×W_tok, H_tok×W_tok)
  e.g. for 1008px: B, 16, 5184, 5184

Usage:
    # requires HF token if model is gated:
    python profile_sam3_hf_image_encoder.py --hf_token hf_...
    python profile_sam3_hf_image_encoder.py  # if already logged in via `huggingface-cli login`

    # pick model variant:
    python profile_sam3_hf_image_encoder.py --model facebook/sam3
    python profile_sam3_hf_image_encoder.py --model facebook/sam3.1

    # custom resolution (model default is 1008):
    python profile_sam3_hf_image_encoder.py --img_size 1008

    # control dtype and repeats:
    python profile_sam3_hf_image_encoder.py --dtype bfloat16 --repeats 20
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Timing primitives
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
    stats.peak_mem_mb = max(stats.peak_mem_mb, max(0, peak - mem_before) / 1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Hook-based sub-module timer
# ─────────────────────────────────────────────────────────────────────────────

class HookTimer:
    """
    Attaches pre/post forward hooks to a module to measure its wall-clock
    execution time with CUDA events. Call `.remove()` to deregister.
    """

    def __init__(self, module: nn.Module, stats: Stats, device: torch.device):
        self.stats  = stats
        self.device = device
        self._start_event: Optional[torch.cuda.Event] = None

        self._pre  = module.register_forward_pre_hook(self._pre_hook)
        self._post = module.register_forward_hook(self._post_hook)

    def _pre_hook(self, module, args):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._mem_before = torch.cuda.memory_allocated(self.device)
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event   = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def _post_hook(self, module, args, output):
        self._end_event.record()
        torch.cuda.synchronize(self.device)
        self.stats.latencies_ms.append(self._start_event.elapsed_time(self._end_event))
        peak = torch.cuda.max_memory_allocated(self.device)
        self.stats.peak_mem_mb = max(
            self.stats.peak_mem_mb,
            max(0, peak - self._mem_before) / 1024 ** 2,
        )

    def remove(self):
        self._pre.remove()
        self._post.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Attention-shape capture hook
# ─────────────────────────────────────────────────────────────────────────────

class AttnShapeCapture:
    """
    Captures the attention map shape from Sam3ViTRoPEAttention.
    Shape is computed analytically from the input tensor + layer config
    (works regardless of attn implementation — eager / SDPA / flash).
    """
    def __init__(self, layer):
        self.shape: Optional[Tuple] = None
        self._hook = layer.attention.register_forward_pre_hook(self._capture)
        self._window_size = layer.window_size
        self._num_heads   = layer.attention.num_attention_heads

    def _capture(self, module, args):
        # args[0] is hidden_states: (B, H, W, C)  (or (nW*B, ws, ws, C) if already windowed)
        x = args[0]
        B, H, W, _ = x.shape
        seq = H * W
        self.shape = (B, self._num_heads, seq, seq)   # (batch_or_windows, heads, seq, seq)

    def remove(self):
        self._hook.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_vision_encoder_pretrained(model_id: str, hf_token: Optional[str], device: torch.device, dtype: torch.dtype):
    """Load real SAM3 weights from HuggingFace (requires access token for gated repo)."""
    from transformers import Sam3Model

    kwargs = dict(torch_dtype=dtype)
    if hf_token:
        kwargs["token"] = hf_token

    print(f"Loading {model_id} ...")
    model = Sam3Model.from_pretrained(model_id, **kwargs)
    vision_enc = model.vision_encoder   # Sam3VisionModel
    del model                           # free the rest
    vision_enc = vision_enc.to(device).eval()
    return vision_enc


def build_mock_vision_encoder(device: torch.device, dtype: torch.dtype):
    """
    Build Sam3VisionModel with random weights from the default SAM3 config.
    No HF token needed — useful for architecture profiling without the checkpoint.
    """
    from transformers import Sam3VisionModel, Sam3VisionConfig, Sam3ViTConfig

    print("Building mock SAM3 vision encoder from default config (random weights) ...")

    # ViT backbone config matching facebook/sam3
    vit_cfg = Sam3ViTConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=32,
        num_attention_heads=16,
        image_size=1008,
        patch_size=14,
        window_size=24,
        global_attn_indexes=[7, 15, 23, 31],
        layer_scale_init_value=1e-6,
    )

    # Vision model config (backbone + FPN neck)
    vis_cfg = Sam3VisionConfig(
        backbone_config=vit_cfg,
        fpn_hidden_size=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
    )

    vision_enc = Sam3VisionModel(vis_cfg).to(dtype).to(device).eval()
    return vision_enc


# ─────────────────────────────────────────────────────────────────────────────
# Profiling
# ─────────────────────────────────────────────────────────────────────────────

INTRA_COMPONENTS = ("embeddings", "layer_norm1", "rotary_emb", "attention", "layer_norm2", "mlp")


@torch.no_grad()
def profile_vision_encoder(
    vision_enc: nn.Module,
    img_size: int,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
):
    backbone = vision_enc.backbone     # Sam3ViTModel
    neck     = vision_enc.neck         # Sam3VisionNeck
    cfg      = backbone.config

    n_layers          = cfg.num_hidden_layers          # 32
    global_idxs       = set(cfg.global_attn_indexes)   # {7, 15, 23, 31}
    window_size       = cfg.window_size                 # 24
    num_heads         = cfg.num_attention_heads         # 16
    patch_size        = cfg.patch_size                  # 14
    H_tok = W_tok     = img_size // patch_size          # 72 for 1008px

    # ── Stats containers ─────────────────────────────────────────────────────
    embed_stats  = Stats()
    final_ln_stats = Stats()
    neck_stats   = Stats()
    backbone_stats = Stats()

    # Per-layer intra-component stats
    layer_stats: List[Dict[str, Stats]] = [
        {c: Stats() for c in ("layer_norm1", "rotary_emb", "attention", "layer_norm2", "mlp")}
        for _ in range(n_layers)
    ]

    # ── Register hooks ────────────────────────────────────────────────────────
    timers: List[HookTimer] = []
    shape_caps: List[AttnShapeCapture] = []

    timers.append(HookTimer(backbone.embeddings, embed_stats, device))
    timers.append(HookTimer(backbone.layer_norm, final_ln_stats, device))
    timers.append(HookTimer(neck, neck_stats, device))
    timers.append(HookTimer(backbone, backbone_stats, device))

    for i, layer in enumerate(backbone.layers):
        s = layer_stats[i]
        timers.append(HookTimer(layer.layer_norm1, s["layer_norm1"], device))
        timers.append(HookTimer(layer.rotary_emb,  s["rotary_emb"],  device))
        timers.append(HookTimer(layer.attention,   s["attention"],   device))
        timers.append(HookTimer(layer.layer_norm2, s["layer_norm2"], device))
        timers.append(HookTimer(layer.mlp,         s["mlp"],         device))
        shape_caps.append(AttnShapeCapture(layer))

    # ── Warm-up ───────────────────────────────────────────────────────────────
    dummy = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)
    for _ in range(2):
        vision_enc(pixel_values=dummy)
    torch.cuda.synchronize(device)

    # ── Timed repeats ─────────────────────────────────────────────────────────
    neck_direct_stats = Stats()    # timed separately to avoid hook double-counting
    for _ in range(repeats):
        x = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)
        vision_enc(pixel_values=x)

    # Capture last attention shapes (stable after warm-up)
    attn_shapes = [cap.shape for cap in shape_caps]

    # Remove all hooks
    for t in timers:
        t.remove()
    for c in shape_caps:
        c.remove()

    # ── Compute totals ────────────────────────────────────────────────────────
    # backbone total = embed + all layers + final_ln  (what backbone_stats measures)
    all_layer_means = [sum(layer_stats[i][c].mean for c in ("layer_norm1", "rotary_emb", "attention", "layer_norm2", "mlp"))
                       for i in range(n_layers)]

    total_mean = backbone_stats.mean + neck_stats.mean

    # ── Print ─────────────────────────────────────────────────────────────────
    W = 86
    print(f"\n{'='*W}")
    print(f"SAM3 HuggingFace Image Encoder Profile")
    print(f"  model: {cfg.__class__.__name__}  |  layers={n_layers}  |  hidden={cfg.hidden_size}  "
          f"|  heads={num_heads}  |  img={img_size}px  |  tokens={H_tok}×{W_tok}={H_tok*W_tok}")
    print(f"  window_size={window_size}  |  global_layers={sorted(global_idxs)}  |  repeats={repeats}")
    print(f"{'='*W}")
    hdr = f"{'Component':<34} {'Mean ms':>9} {'Min ms':>9} {'Max ms':>9} {'Mem MB':>9} {'Share%':>8}"
    print(hdr)
    print("─" * W)

    def row(name, s: Stats, indent=0):
        pct = 100 * s.mean / total_mean if total_mean else 0
        pad = "  " * indent
        print(f"{pad}{name:<{34 - 2*indent}} {s.mean:>9.2f} {s.mn:>9.2f} {s.mx:>9.2f} "
              f"{s.peak_mem_mb:>9.1f} {pct:>7.1f}%")

    # ── Backbone top-level ────────────────────────────────────────────────────
    row("Backbone (total)", backbone_stats)
    row("  embeddings", embed_stats, indent=1)
    print()

    # ── Per-layer breakdown ───────────────────────────────────────────────────
    # Aggregate window layers and global layers separately for summary
    win_idx  = [i for i in range(n_layers) if i not in global_idxs]
    glob_idx = [i for i in range(n_layers) if i in global_idxs]

    def _agg(indices, comp):
        vals = [layer_stats[i][comp].mean for i in indices]
        mns  = [layer_stats[i][comp].mn   for i in indices]
        mxs  = [layer_stats[i][comp].mx   for i in indices]
        return sum(vals), sum(mns), sum(mxs), sum(vals) / len(vals) if vals else 0

    def agg_stats(indices, comp) -> Stats:
        s = Stats()
        total_v = sum(layer_stats[i][comp].mean for i in indices)
        total_n = sum(layer_stats[i][comp].mn   for i in indices)
        total_x = sum(layer_stats[i][comp].mx   for i in indices)
        s.latencies_ms = [total_v]   # single "sum" value for display
        # use mn/mx as sum-of-mins / sum-of-maxs
        s._mn_override = total_n
        s._mx_override = total_x
        return s

    def print_layer_group(name, indices, example_idx):
        """Print timing for one group of layers (window or global)."""
        group_mean = sum(all_layer_means[i] for i in indices)
        group_mn   = sum(min(layer_stats[i][c].mn for c in ("layer_norm1","rotary_emb","attention","layer_norm2","mlp"))
                        for i in indices)  # not really useful but keep shape
        pct = 100 * group_mean / total_mean if total_mean else 0
        avg_per_layer = group_mean / len(indices) if indices else 0

        # Attention map shape for the first layer in this group
        shape = attn_shapes[example_idx] if example_idx < len(attn_shapes) else None
        shape_str = f"  attn_map={shape}" if shape else ""

        print(f"  {name:<32} {group_mean:>9.2f}  (avg/layer={avg_per_layer:.2f}ms){shape_str}   {pct:.1f}%")

        # per-component aggregate (sum across all layers in group)
        for comp in ("layer_norm1", "rotary_emb", "attention", "layer_norm2", "mlp"):
            comp_sum  = sum(layer_stats[i][comp].mean for i in indices)
            comp_mn   = sum(layer_stats[i][comp].mn   for i in indices)
            comp_mx   = sum(layer_stats[i][comp].mx   for i in indices)
            comp_avg  = comp_sum / len(indices) if indices else 0
            cpct      = 100 * comp_sum / total_mean if total_mean else 0
            print(f"    {'└─ ' + comp:<30} {comp_sum:>9.2f} {comp_mn:>9.2f} {comp_mx:>9.2f}"
                  f"            {cpct:>7.1f}%  (avg/layer={comp_avg:.3f}ms)")

    print_layer_group(
        f"Window-attn layers  [{len(win_idx)} layers, ws={window_size}]",
        win_idx, win_idx[0] if win_idx else 0
    )
    print()
    print_layer_group(
        f"Global-attn layers  [{len(glob_idx)} layers]",
        glob_idx, glob_idx[0] if glob_idx else 0
    )
    print()
    row("  final layer_norm", final_ln_stats, indent=1)
    print()

    # ── Neck ──────────────────────────────────────────────────────────────────
    row("FPN Neck (total)", neck_stats)
    for i, fpn_layer in enumerate(neck.fpn_layers):
        sf = fpn_layer.scale_factor
        fpn_s = Stats()
        fpn_timer = HookTimer(fpn_layer, fpn_s, device)
        # Quick re-run to get per-layer neck timing
        dummy = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)
        for _ in range(repeats):
            vision_enc(pixel_values=dummy)
        fpn_timer.remove()
        h_out = int(H_tok * sf)
        w_out = int(W_tok * sf)
        pct   = 100 * fpn_s.mean / total_mean if total_mean else 0
        print(f"  {'└─ FPN layer ' + str(i) + f'  (×{sf})':<32} {fpn_s.mean:>9.2f} {fpn_s.mn:>9.2f} "
              f"{fpn_s.mx:>9.2f} {fpn_s.peak_mem_mb:>9.1f} {pct:>7.1f}%  → {h_out}×{w_out} tokens")

    print("─" * W)
    print(f"{'TOTAL':<34} {total_mean:>9.2f}")
    print(f"\nEstimated throughput: {1000 / total_mean:.2f} fps")

    # ── Attention map shape summary ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Attention map shapes  (batch=1):")
    print(f"{'─'*60}")

    n_windows = (H_tok // window_size) * (W_tok // window_size)
    win_seq   = window_size * window_size
    glob_seq  = H_tok * W_tok

    win_attn_shape  = (1 * n_windows, num_heads, win_seq,  win_seq)
    glob_attn_shape = (1,             num_heads, glob_seq, glob_seq)

    win_bytes  = 2 * win_seq  * win_seq  * n_windows * num_heads / 1024**2   # fp16
    glob_bytes = 2 * glob_seq * glob_seq             * num_heads / 1024**2

    print(f"  Window-attn layers  ({len(win_idx)} layers):  "
          f"{win_attn_shape}  ≈ {win_bytes:.1f} MB/layer")
    print(f"    n_windows = ({H_tok}/{window_size}) × ({W_tok}/{window_size}) = {n_windows}")
    print(f"    seq per window = {window_size}² = {win_seq}")
    print()
    print(f"  Global-attn layers  ({len(glob_idx)} layers):  "
          f"{glob_attn_shape}  ≈ {glob_bytes:.1f} MB/layer")
    print(f"    seq = {H_tok}×{W_tok} = {glob_seq}  ← target for block-sparse attention")

    # ── Window vs global per-component timing ─────────────────────────────────
    print(f"\n{'─'*60}")
    print("Window vs Global attention timing comparison (avg per layer):")
    print(f"  {'Component':<20} {'Window (ms)':>14} {'Global (ms)':>14}  Ratio")
    print(f"{'─'*60}")
    for comp in ("layer_norm1", "rotary_emb", "attention", "layer_norm2", "mlp"):
        w_avg = sum(layer_stats[i][comp].mean for i in win_idx)  / max(len(win_idx), 1)
        g_avg = sum(layer_stats[i][comp].mean for i in glob_idx) / max(len(glob_idx), 1)
        ratio = g_avg / w_avg if w_avg > 0 else float("nan")
        print(f"  {comp:<20} {w_avg:>14.3f} {g_avg:>14.3f}  {ratio:>5.1f}×")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Profile SAM3 HuggingFace image encoder")
    p.add_argument("--model",    default="facebook/sam3",
                   help="HuggingFace model ID (facebook/sam3 or facebook/sam3.1)")
    p.add_argument("--hf_token", default=None,
                   help="HuggingFace access token (needed for gated repo). "
                        "Alternatively run: huggingface-cli login")
    p.add_argument("--mock",     action="store_true",
                   help="Use random-weight model built from default SAM3 config. "
                        "No HF token needed — good for architecture/latency analysis.")
    p.add_argument("--img_size", type=int, default=1008,
                   help="Input image size in pixels (model default: 1008)")
    p.add_argument("--repeats",  type=int, default=10)
    p.add_argument("--device",   default="cuda:0")
    p.add_argument("--dtype",    default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    dtype  = {"float16": torch.float16,
               "bfloat16": torch.bfloat16,
               "float32":  torch.float32}[args.dtype]

    print(f"device={device}  dtype={dtype}  img_size={args.img_size}")

    if args.mock:
        vision_enc = build_mock_vision_encoder(device, dtype)
    else:
        try:
            vision_enc = load_vision_encoder_pretrained(args.model, args.hf_token, device, dtype)
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e):
                print("\n[ERROR] Model is gated. To access it:")
                print("  1. Request access at https://huggingface.co/facebook/sam3")
                print("  2. Then pass your token:  --hf_token hf_...")
                print("     or run:                 huggingface-cli login")
                print("\n  To profile the architecture without weights, use:  --mock")
                raise SystemExit(1)
            raise

    with torch.no_grad():
        profile_vision_encoder(vision_enc, args.img_size, args.repeats, device, dtype)


if __name__ == "__main__":
    main()
