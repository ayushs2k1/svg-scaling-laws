# Scaling Laws for SVG Language Models

CS-GY 6923 optional project (Spring 2026, NYU Tandon).  Trains decoder-only
Transformers at five parameter scales on SVG (Scalable Vector Graphics) code,
derives power-law scaling exponents, compares standard parameterisation (SP)
with Maximal Update Parameterisation (µP), and generates/evaluates SVG samples.

## Repository layout

```
notebooks/svg_scaling_laws.ipynb   # single self-contained notebook (all five parts)
requirements.txt                   # Python package dependencies
```

## Quickstart (Google Colab)

Open `notebooks/svg_scaling_laws.ipynb` in Colab, set runtime to **GPU**,
then run cells top-to-bottom.  Each training cell is resumable: if a
`_final.pt` checkpoint already exists in the Drive folder it loads that
instead of re-training.

> All heavy training is done inside the notebook.  No separate Python
> modules or shell pipelines are needed.

## Key design decisions

| Choice | Rationale |
|--------|-----------|
| BPE vocab 4 096 | Small enough for tiny models; large enough to capture SVG keywords |
| Context length 512 | Covers ~95% of cleaned SVGs; keeps memory manageable for XL |
| Weight tying (SP only) | Standard LM practice; disabled under µP (incompatible with MuReadout) |
| Cosine LR + warmup 200 | Empirically stable across all model sizes |
| 98/1/1 split | Split by file, not position, to prevent token-level leakage |

## What is from nanoGPT vs. original

The causal self-attention block, pre-LayerNorm residual structure, and weight
initialisation scheme follow nanoGPT (Karpathy, 2022).  Everything else —
µP integration, SVG preprocessing, BPE tokenisation pipeline, binary packing,
scaling-law fitting, generation, and evaluation — is written from scratch for
this project.

## References

- Kaplan et al. (2020) — *Scaling Laws for Neural Language Models*
- Hoffmann et al. (2022) — *Training Compute-Optimal Large Language Models*
- Yang et al. (2022) — *Tensor Programs V: Zero-Shot Hyperparameter Transfer* (µP)
- Rodriguez et al. (2023) — *StarVector: Generating SVG Code from Images and Text*
- Karpathy (2022) — [nanoGPT](https://github.com/karpathy/nanoGPT)
