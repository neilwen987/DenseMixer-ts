import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator, PercentFormatter, FuncFormatter, NullLocator, NullFormatter
import re

def apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 600,
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": False,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.6,
        }
    )

def to_tokens_by_expert(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2:  # [T, E]
        return t
    if t.dim() == 3:  # [B, S, E]
        return t.reshape(-1, t.size(-1))  # [T, E]
    raise ValueError(f"Unsupported tensor shape {tuple(t.shape)}. Expect [T,E] or [B,S,E].")

def parse_layer_index_from_key(key: Any) -> Optional[int]:
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        try:
            return int(key)
        except Exception:
            nums = re.findall(r"\d+", key)
            if nums:
                try:
                    return int(nums[-1])
                except Exception:
                    return None
    return None

def find_key_for_index(cache: Dict[Any, torch.Tensor], desired_index_zero_based: int) -> Optional[Any]:
    for k in cache.keys():
        idx = parse_layer_index_from_key(k)
        if idx is not None and idx == desired_index_zero_based:
            return k
    return None

def compute_proportions(value: torch.Tensor, aggregate: str = "count", threshold: float = 0.0, topk: int = None) -> np.ndarray:
    t_te = to_tokens_by_expert(value)  # [T, E]
    num_experts = t_te.shape[1]

    if aggregate == "weight":
        load_per_expert = t_te.sum(dim=0)
    else:  # count
        if topk is not None:
            k_sel = min(topk, num_experts)
            topk_idx = torch.topk(t_te, k=k_sel, dim=1).indices  # [T, k]
            counts = torch.bincount(topk_idx.reshape(-1), minlength=num_experts)
            load_per_expert = counts.to(dtype=torch.float32)
        else:
            load_per_expert = (t_te > threshold).sum(dim=0).to(dtype=torch.float32)

    total = load_per_expert.sum()
    if total == 0:
        return np.zeros(num_experts, dtype=np.float32)
    p = (load_per_expert / total).cpu().numpy()
    return p

def plot_histograms(
    props_topk: List[np.ndarray],
    props_seqtopk: List[np.ndarray],
    layers: List[str],
    out_path: Path,
    bins: int = 20,
) -> None:
    apply_nature_style()
    n = len(layers)
    fig_height = 1.5 * n + 1.5
    fig, axs = plt.subplots(n, 1, figsize=(3.5, fig_height), sharex=True)
    if n == 1:
        axs = [axs]

    colors = {"topk": "#0072B2", "seqtopk": "#D55E00"}
    bar_width = 0.42

    for i, (ax, layer, p_t, p_s) in enumerate(zip(axs, layers, props_topk, props_seqtopk)):
        E = len(p_t)
        x = np.arange(1, E + 1)

        ax.bar(x - bar_width / 2, p_t, width=bar_width, label="TopK", color=colors["topk"], alpha=0.9)
        ax.bar(x + bar_width / 2, p_s, width=bar_width, label="SeqTopK", color=colors["seqtopk"], alpha=0.9)

        # Sparse ticks to avoid clutter
        step = max(1, int(np.ceil(E / 8)))
        tick_positions = list(range(1, E + 1, step))
        if tick_positions[-1] != E:
            tick_positions.append(E)
        ax.set_xticks(tick_positions)

        ymax = float(max(p_t.max(initial=0.0), p_s.max(initial=0.0)))
        ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)

        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_ylabel("Token proportion (%)")
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="0.85")

    axs[-1].set_xlabel("Expert id")

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path.as_posix(), bbox_inches="tight")

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot histograms of expert utilization for selected layers.")
    parser.add_argument("--topk_pt", type=Path, default="/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/llama-factory/eval/olmoe_results/gsm/load_balance/topk/routing_prompt_only.pt", help="Path to topk routing cache pt file")
    parser.add_argument("--seqtopk_pt", type=Path, default="/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/llama-factory/eval/olmoe_results/gsm/load_balance/sptopk/routing_prompt_only.pt", help="Path to seqtopk routing cache pt file")
    parser.add_argument("--layers", type=str, default="1,7,16", help="Comma-separated layer indices (1-based, e.g., 1,7,16)")
    parser.add_argument("--aggregate", type=str, choices=["weight", "count"], default="count", help="Aggregation method")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for count mode")
    parser.add_argument("--topk", type=int, default=None, help="TopK for count mode if applicable")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    parser.add_argument("--out", type=Path, default="expert_utilization_hist.svg", help="Output image path")
    parser.add_argument("--legend_xy", type=str, default="0.55,1.02", help="Legend anchor in axes coords 'x,y' (0-1)")
    parser.add_argument("--legend_fontsize", type=float, default=8.0, help="Legend font size")
    parser.add_argument("--legend_data_xy", type=str, default=None, help="Legend anchor in data coords 'x,y' (e.g., '49,0.12')")
    parser.add_argument("--legend_loc", type=str, default="upper left", help="Legend loc when anchoring (e.g., 'center','upper left')")
    args = parser.parse_args()

    cache_topk = torch.load(args.topk_pt, map_location="cpu")
    cache_seqtopk = torch.load(args.seqtopk_pt, map_location="cpu")

    layer_indices_1based = [int(x) for x in args.layers.split(",")]
    desired_zero_based = [i - 1 for i in layer_indices_1based]

    props_topk: List[np.ndarray] = []
    props_seqtopk: List[np.ndarray] = []
    display_layers: List[str] = []

    for shown_label, idx0 in zip(layer_indices_1based, desired_zero_based):
        key_topk = find_key_for_index(cache_topk, idx0)
        key_seq = find_key_for_index(cache_seqtopk, idx0)
        if key_topk is None or key_seq is None:
            print(f"Warning: layer {shown_label} not found in one of caches. topk_has={key_topk is not None}, seqtopk_has={key_seq is not None}. Skipped.")
            continue
        p_t = compute_proportions(cache_topk[key_topk], args.aggregate, args.threshold, args.topk)
        p_s = compute_proportions(cache_seqtopk[key_seq], args.aggregate, args.threshold, args.topk)
        props_topk.append(p_t)
        props_seqtopk.append(p_s)
        display_layers.append(str(shown_label))

    if not props_topk:
        available_topk = [parse_layer_index_from_key(k) for k in cache_topk.keys()]
        available_seq = [parse_layer_index_from_key(k) for k in cache_seqtopk.keys()]
        raise ValueError(f"No valid layers found. Available (topk): {available_topk}; (seqtopk): {available_seq}")

    # Save one image per layer
    base_out: Path = args.out
    parent = base_out.parent
    stem = base_out.stem
    suffix = base_out.suffix if base_out.suffix else ".png"

    for layer, p_t, p_s in zip(display_layers, props_topk, props_seqtopk):
        apply_nature_style()
        fig, ax = plt.subplots(figsize=(3.5, 2.0))

        E = len(p_t)
        x = np.arange(1, E + 1)
        bar_width = 0.42

        ax.bar(x - bar_width / 2, p_t, width=bar_width, label="TopK", color="#0072B2", alpha=0.9)
        ax.bar(x + bar_width / 2, p_s, width=bar_width, label="SeqTopK", color="#D55E00", alpha=0.9)

        # Sparse x ticks
        step = max(1, int(np.ceil(E / 8)))
        tick_positions = list(range(1, E + 1, step))
        if tick_positions[-1] != E:
            tick_positions.append(E)
        ax.set_xticks(tick_positions)

        ymax = float(max(p_t.max(initial=0.0), p_s.max(initial=0.0)))
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)

        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_xlabel("Expert id")
        ax.set_ylabel("Token proportion (%)")

        # Show y as percentage numbers without the % sign and hide minor ticks
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.1f}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Legend placement: prefer data coordinates if provided
        if args.legend_data_xy is not None:
            try:
                dx, dy = (float(s) for s in str(args.legend_data_xy).split(","))
                ax.legend(loc=args.legend_loc, bbox_to_anchor=(dx, dy), bbox_transform=ax.transData, fontsize=args.legend_fontsize, frameon=False)
            except Exception:
                try:
                    lx, ly = (float(s) for s in str(args.legend_xy).split(","))
                except Exception:
                    lx, ly = 0.5, 0.5
                ax.legend(loc="upper left", bbox_to_anchor=(lx, ly), fontsize=args.legend_fontsize, frameon=False)
        else:
            try:
                lx, ly = (float(s) for s in str(args.legend_xy).split(","))
            except Exception:
                lx, ly = 0.5, 0.5
            ax.legend(loc="upper left", bbox_to_anchor=(lx, ly), fontsize=args.legend_fontsize, frameon=False)

        fig.tight_layout(pad=0.5)
        out_path = parent / f"{stem}_layer{layer}{suffix}"
        fig.savefig(out_path.as_posix(), bbox_inches="tight",transparent=True, dpi = 600)
        try:
            plt.close(fig)
        except Exception:
            pass

if __name__ == "__main__":
    main()
