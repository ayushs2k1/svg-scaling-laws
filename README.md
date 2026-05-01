# Scaling Laws for SVG Language Models

CS-GY 6923 optional project (Spring 2026, NYU Tandon).  Trains decoder-only
Transformers at five parameter scales on SVG (Scalable Vector Graphics) code,
derives power-law scaling exponents, compares standard parameterisation (SP)
with Maximal Update Parameterisation (µP), and generates/evaluates SVG samples.

## Repository layout

```
notebooks/svg_scaling_laws.ipynb   # single self-contained notebook (all five parts)
requirements.txt                   # Python package dependencies
report/report.tex                  # LaTeX source for the project report
report/report.pdf                  # Compiled PDF report
report/*.png                       # Figures embedded in the report
svg_scaling_results/               # Output from the final notebook run
  final_summary.json               #   Perplexity, generation metrics, scaling-law fits
  sp_results.json                  #   Per-size SP training stats
  mup_results.json                 #   Per-size µP training stats
  *.png                            #   Training curves, LR sweeps, scaling plots
  samples/                         #   Generated SVG files (30 unconditional + 10 prefix)
```

## Key results

| Metric | Value |
|--------|-------|
| Corpus size | 182,720 cleaned SVGs · ~100 M tokens |
| Model scales | Tiny (1.4 M) · Small (3.5 M) · Medium (12.4 M) · Large (33.8 M) · XL (88.5 M) |
| Best LR (SP & µP) | 3.16 × 10⁻³ |
| SP outcome | Tiny/Small/Medium converge; Large & XL diverge (LR too high at scale) |
| µP outcome | All five sizes converge; val loss 0.550 → 0.480 (Tiny → Large) |
| Scaling exponent (µP) | α = 1.01 · floor c = 0.486 · CI ±0.040 at 916 M-param extrapolation |
| Test perplexity (µP-XL) | 2.62 |
| XML validity (30 samples) | 6.7 % |
| Structural validity | 23.3 % (7× improvement after switching to µP model + raising token budget) |

## Quickstart (Google Colab)

1. Open `notebooks/svg_scaling_laws.ipynb` in Colab and set runtime to **GPU**.
2. Fill in your HuggingFace token in the HF Hub cell (cell after setup) so that
   checkpoints persist across disconnects.
3. Run cells top-to-bottom.  Every training cell is resumable: if a `_final.pt`
   checkpoint already exists in `/content/checkpoints` (or on HF Hub), it is
   loaded instead of re-trained.

> All heavy training is done inside the notebook.  No separate Python
> modules or shell pipelines are needed.

## Checkpoint persistence

Completed model checkpoints are pushed to a private HuggingFace Hub repository
(`ayush2k1/svg-scaling-ckpts`) immediately after each model finishes.  On
reconnect, `hf_pull_all()` at the top of each training cell restores them
automatically and skips re-training.

## Key design decisions

| Choice | Rationale |
|--------|-----------|
| BPE vocab 4 096 | Small enough for tiny models; large enough to capture SVG keywords |
| Context length 512 | Covers the majority of cleaned SVGs; keeps memory manageable for XL |
| Weight tying (SP only) | Standard LM practice; disabled under µP (incompatible with MuReadout) |
| Cosine LR + 200-step warmup | Empirically stable across all model sizes in both SP and µP |
| 98/1/1 split by document | Prevents token-level leakage between train and val/test |
| µP-XL for generation | µP models converge stably at all scales; SP-XL diverged |
| max_new = 600 tokens | Median SVG requires ~450–500 tokens; 600 allows closing </svg> |

## What is from nanoGPT vs. original

The causal self-attention block, pre-LayerNorm residual structure, and weight
initialisation scheme follow nanoGPT (Karpathy, 2022).  Everything else —
µP integration, SVG preprocessing, BPE tokenisation pipeline, binary packing,
HuggingFace Hub checkpoint persistence, scaling-law fitting with delta-method
confidence intervals, generation, and evaluation — is written from scratch for
this project.

## References

- Kaplan et al. (2020) — *Scaling Laws for Neural Language Models*
- Hoffmann et al. (2022) — *Training Compute-Optimal Large Language Models*
- Yang et al. (2022) — *Tensor Programs V: Zero-Shot Hyperparameter Transfer* (µP)
- Rodriguez et al. (2023) — *StarVector: Generating SVG Code from Images and Text*
- Karpathy (2022) — [nanoGPT](https://github.com/karpathy/nanoGPT)
- Dao et al. (2022) — *FlashAttention: Fast and Memory-Efficient Exact Attention*
