"""
Profile each SAM2/SAM3 component during video segmentation inference.

Components profiled:
  1. image_encoder      — Hiera backbone (per-frame)
  2. memory_attention   — cross-attention between current frame & memory bank
  3. prompt_encoder     — encodes click/box/mask prompts
  4. mask_decoder       — predicts masks from image + memory features
  5. memory_encoder     — encodes predicted mask + image features into memory

Usage:
    python profile_sam3_video.py \
        --checkpoint /path/to/sam2_hiera_large.pt \
        --model_cfg   sam2_hiera_l.yaml \
        --video_dir   /path/to/frames/ \
        --num_frames  16 \
        --repeats     10

If --mock is passed, synthetic tensors are used (no real checkpoint needed).
"""

import argparse
import time
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@dataclass
class ComponentStats:
    name: str
    latencies_ms: List[float] = field(default_factory=list)
    peak_mem_mb: float = 0.0

    @property
    def mean_ms(self):
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def min_ms(self):
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self):
        return max(self.latencies_ms) if self.latencies_ms else 0.0


@contextmanager
def cuda_timer(stats: ComponentStats, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize(device)
    elapsed_ms = start.elapsed_time(end)
    peak = torch.cuda.max_memory_allocated(device)
    stats.latencies_ms.append(elapsed_ms)
    stats.peak_mem_mb = max(stats.peak_mem_mb, max(0, peak - mem_before) / (1024 ** 2))


# ---------------------------------------------------------------------------
# Mock SAM2-like components (used when --mock or no checkpoint)
# ---------------------------------------------------------------------------

class MockImageEncoder(nn.Module):
    """Approximates Hiera-L: ViT-style backbone producing multi-scale features."""
    def __init__(self, img_size=1024, embed_dim=1280, num_heads=16, depth=32, dtype=torch.float16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16).to(dtype)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, dtype=dtype
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.img_size = img_size
        self.seq_len = (img_size // 16) ** 2  # 4096 for 1024x1024

    def forward(self, x):
        # x: (B, 3, H, W)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, seq, C)
        return self.encoder(tokens)


class MockMemoryAttention(nn.Module):
    """Cross-attention between current frame tokens and memory bank tokens."""
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, dtype=torch.float16):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 8,
            batch_first=True, dtype=dtype
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, curr_tokens, memory_tokens):
        return self.decoder(curr_tokens, memory_tokens)


class MockPromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, dtype=torch.float16):
        super().__init__()
        self.point_embed = nn.Embedding(2, embed_dim).to(dtype)  # fg/bg labels

    def forward(self, points, labels):
        # points: (B, N, 2), labels: (B, N)
        return self.point_embed(labels)  # (B, N, embed_dim)


class MockMaskDecoder(nn.Module):
    """Lightweight two-way transformer + upsampling head."""
    def __init__(self, embed_dim=256, num_heads=8, num_multimask=3, dtype=torch.float16):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 8,
            batch_first=True, dtype=dtype
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=2)
        self.iou_head = nn.Linear(embed_dim, num_multimask + 1).to(dtype)
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, num_multimask + 1, 1),
        ).to(dtype)

    def forward(self, image_embed, prompt_embed):
        # image_embed: (B, seq, C), prompt_embed: (B, N, C)
        fused = self.transformer(prompt_embed, image_embed)  # (B, N, C)
        iou_pred = self.iou_head(fused[:, 0])
        B, seq, C = image_embed.shape
        h = w = int(math.isqrt(seq))
        feat_map = image_embed.transpose(1, 2).reshape(B, C, h, w)
        masks = self.mask_head(feat_map.contiguous())
        return masks, iou_pred


class MockMemoryEncoder(nn.Module):
    """Encodes predicted mask + image features into a compact memory token."""
    def __init__(self, embed_dim=256, out_dim=64, dtype=torch.float16):
        super().__init__()
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, out_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        ).to(dtype)
        self.fuse_proj = nn.Linear(embed_dim + out_dim, embed_dim).to(dtype)

    def forward(self, image_embed, mask):
        # image_embed: (B, seq, C), mask: (B, 1, H, W)
        B, seq, C = image_embed.shape
        h = w = int(math.isqrt(seq))
        mask_feat = self.mask_conv(mask)                          # (B, out_dim, H, W)
        mask_feat = nn.functional.adaptive_avg_pool2d(mask_feat, (h, w))
        mask_feat = mask_feat.flatten(2).transpose(1, 2)         # (B, seq, out_dim)
        fused = torch.cat([image_embed, mask_feat], dim=-1)
        return self.fuse_proj(fused)                              # (B, seq, C)


# ---------------------------------------------------------------------------
# Profile runner
# ---------------------------------------------------------------------------

def build_mock_model(device, dtype):
    """Build lightweight mock models — heavier Hiera depth for realism."""
    models = {
        "image_encoder":   MockImageEncoder(img_size=1024, embed_dim=1280, num_heads=16, depth=4, dtype=dtype).to(device).eval(),
        "memory_attention": MockMemoryAttention(embed_dim=256, num_heads=8, num_layers=4, dtype=dtype).to(device).eval(),
        "prompt_encoder":  MockPromptEncoder(embed_dim=256, dtype=dtype).to(device).eval(),
        "mask_decoder":    MockMaskDecoder(embed_dim=256, num_heads=8, dtype=dtype).to(device).eval(),
        "memory_encoder":  MockMemoryEncoder(embed_dim=256, out_dim=64, dtype=dtype).to(device).eval(),
    }
    return models


def build_sam2_model(checkpoint, model_cfg, device):
    """Load real SAM2 model. Requires sam2 package installed."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        sam2 = predictor.model
        sam2.eval()
        components = {
            "image_encoder":    sam2.image_encoder,
            "memory_attention": sam2.memory_attention,
            "prompt_encoder":   sam2.sam_prompt_encoder,
            "mask_decoder":     sam2.sam_mask_decoder,
            "memory_encoder":   sam2.memory_encoder,
        }
        return components
    except ImportError:
        raise ImportError("sam2 package not found. Install from https://github.com/facebookresearch/segment-anything-2 or use --mock.")


@torch.no_grad()
def profile_components(
    components: Dict[str, nn.Module],
    num_frames: int,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
    img_size: int = 1024,
    embed_dim: int = 256,
    memory_bank_size: int = 7,
    num_points: int = 1,
):
    batch = 1
    seq_len = (img_size // 16) ** 2       # 4096
    mem_seq = memory_bank_size * seq_len  # memory bank sequence length

    all_stats: Dict[str, ComponentStats] = {k: ComponentStats(k) for k in components}

    print(f"\nProfileing {num_frames} frames  |  {repeats} repeats  |  seq_len={seq_len}  |  mem_bank={memory_bank_size} frames")
    print(f"{'Component':<22} {'Mean (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12} {'Peak Mem (MB)':>16}")
    print("-" * 76)

    # Warm-up pass (no timing)
    _warmup(components, batch, seq_len, mem_seq, embed_dim, num_points, img_size, device, dtype)

    for rep in range(repeats):
        # ── 1. Image Encoder ─────────────────────────────────────────────────
        frames = torch.randn(batch, 3, img_size, img_size, device=device, dtype=dtype)
        with cuda_timer(all_stats["image_encoder"], device):
            img_embed = components["image_encoder"](frames)
        # Flatten/project to embed_dim if needed (mock returns 1280-dim)
        if img_embed.shape[-1] != embed_dim:
            img_embed = img_embed[..., :embed_dim].contiguous()

        # ── 2. Memory Attention ───────────────────────────────────────────────
        memory_tokens = torch.randn(batch, mem_seq, embed_dim, device=device, dtype=dtype)
        with cuda_timer(all_stats["memory_attention"], device):
            fused_embed = components["memory_attention"](img_embed, memory_tokens)

        # ── 3. Prompt Encoder ─────────────────────────────────────────────────
        points = torch.randint(0, img_size, (batch, num_points, 2), device=device)
        labels = torch.ones(batch, num_points, device=device, dtype=torch.long)
        with cuda_timer(all_stats["prompt_encoder"], device):
            prompt_embed = components["prompt_encoder"](points, labels)

        # ── 4. Mask Decoder ───────────────────────────────────────────────────
        with cuda_timer(all_stats["mask_decoder"], device):
            masks, iou_pred = components["mask_decoder"](fused_embed, prompt_embed)

        # ── 5. Memory Encoder ─────────────────────────────────────────────────
        best_mask = masks[:, :1]  # take first mask output
        mask_resized = torch.nn.functional.interpolate(
            best_mask.float(), size=(img_size // 16, img_size // 16)
        ).to(dtype)
        with cuda_timer(all_stats["memory_encoder"], device):
            memory_out = components["memory_encoder"](fused_embed, mask_resized)

    # ── Print table ────────────────────────────────────────────────────────
    total_mean = 0.0
    for name, stats in all_stats.items():
        print(f"{name:<22} {stats.mean_ms:>12.2f} {stats.min_ms:>12.2f} {stats.max_ms:>12.2f} {stats.peak_mem_mb:>16.1f}")
        total_mean += stats.mean_ms

    print("-" * 76)
    print(f"{'TOTAL per frame':<22} {total_mean:>12.2f}")
    print(f"\nEstimated throughput: {1000.0 / total_mean:.2f} frames/sec")

    # ── Per-component share ───────────────────────────────────────────────
    print(f"\n{'Component':<22} {'Share %':>10}")
    print("-" * 34)
    for name, stats in all_stats.items():
        pct = 100.0 * stats.mean_ms / total_mean if total_mean > 0 else 0.0
        bar = "#" * int(pct / 2)
        print(f"{name:<22} {pct:>9.1f}%  {bar}")

    return all_stats


@torch.no_grad()
def _warmup(components, batch, seq_len, mem_seq, embed_dim, num_points, img_size, device, dtype):
    """Single silent pass to warm up CUDA kernels."""
    frames = torch.randn(batch, 3, img_size, img_size, device=device, dtype=dtype)
    img_embed = components["image_encoder"](frames)
    if img_embed.shape[-1] != embed_dim:
        img_embed = img_embed[..., :embed_dim].contiguous()
    memory_tokens = torch.randn(batch, mem_seq, embed_dim, device=device, dtype=dtype)
    fused = components["memory_attention"](img_embed, memory_tokens)
    pts = torch.randint(0, img_size, (batch, num_points, 2), device=device)
    labs = torch.ones(batch, num_points, device=device, dtype=torch.long)
    pe = components["prompt_encoder"](pts, labs)
    masks, _ = components["mask_decoder"](fused, pe)
    best_mask = nn.functional.interpolate(masks[:, :1].float(), size=(img_size // 16, img_size // 16)).to(dtype)
    components["memory_encoder"](fused, best_mask)
    torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Optional: torch.profiler trace for detailed GPU kernel breakdown
# ---------------------------------------------------------------------------

@torch.no_grad()
def trace_with_profiler(components, device, dtype, img_size=1024, embed_dim=256, memory_bank_size=7, output_dir="./sam3_profile_trace"):
    import torch.profiler as profiler

    batch = 1
    seq_len = (img_size // 16) ** 2
    mem_seq = memory_bank_size * seq_len

    _warmup(components, batch, seq_len, mem_seq, embed_dim, 1, img_size, device, dtype)

    activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
    schedule = profiler.schedule(wait=1, warmup=1, active=3)

    with profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for step in range(5):
            frames = torch.randn(batch, 3, img_size, img_size, device=device, dtype=dtype)
            with torch.profiler.record_function("image_encoder"):
                img_embed = components["image_encoder"](frames)
            if img_embed.shape[-1] != embed_dim:
                img_embed = img_embed[..., :embed_dim].contiguous()

            mem_tokens = torch.randn(batch, mem_seq, embed_dim, device=device, dtype=dtype)
            with torch.profiler.record_function("memory_attention"):
                fused = components["memory_attention"](img_embed, mem_tokens)

            pts = torch.randint(0, img_size, (batch, 1, 2), device=device)
            labs = torch.ones(batch, 1, device=device, dtype=torch.long)
            with torch.profiler.record_function("prompt_encoder"):
                pe = components["prompt_encoder"](pts, labs)

            with torch.profiler.record_function("mask_decoder"):
                masks, iou = components["mask_decoder"](fused, pe)

            best_mask = nn.functional.interpolate(masks[:, :1].float(), size=(img_size // 16, img_size // 16)).to(dtype)
            with torch.profiler.record_function("memory_encoder"):
                components["memory_encoder"](fused, best_mask)

            prof.step()

    print(f"\nTorch profiler trace saved to: {output_dir}")
    print("\nTop CUDA kernels by self-time:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Profile SAM2/SAM3 components for video segmentation")
    p.add_argument("--checkpoint",   default=None,  help="Path to SAM2 checkpoint (.pt)")
    p.add_argument("--model_cfg",    default=None,  help="SAM2 model config yaml (e.g. sam2_hiera_l.yaml)")
    p.add_argument("--mock",         action="store_true", help="Use mock (synthetic) model — no checkpoint needed")
    p.add_argument("--num_frames",   type=int, default=16)
    p.add_argument("--repeats",      type=int, default=10, help="Timing repeats per component")
    p.add_argument("--img_size",     type=int, default=1024)
    p.add_argument("--embed_dim",    type=int, default=256)
    p.add_argument("--memory_bank",  type=int, default=7,  help="Number of past frames in memory bank")
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--dtype",        default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--trace",        action="store_true", help="Also save a torch.profiler TensorBoard trace")
    p.add_argument("--trace_dir",    default="./sam3_profile_trace")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Device: {device}  |  dtype: {dtype}  |  img_size: {args.img_size}")

    if args.mock or (args.checkpoint is None):
        print("Using MOCK components (--mock or no --checkpoint supplied)")
        components = build_mock_model(device, dtype)
    else:
        print(f"Loading SAM2 from {args.checkpoint}")
        components = build_sam2_model(args.checkpoint, args.model_cfg, device)

    profile_components(
        components,
        num_frames=args.num_frames,
        repeats=args.repeats,
        device=device,
        dtype=dtype,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        memory_bank_size=args.memory_bank,
    )

    if args.trace:
        print("\nRunning torch.profiler trace...")
        trace_with_profiler(components, device, dtype, args.img_size, args.embed_dim, args.memory_bank, args.trace_dir)


if __name__ == "__main__":
    main()
