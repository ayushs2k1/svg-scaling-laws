"""Read results/* JSON and emit LaTeX fragments the report \\input{}s.

Outputs (under report/_auto/):
  data_stats.tex
  arch_table.tex
  results_table.tex
  sweep_sp.tex
  sweep_mup.tex
  scaling_summary.tex
  eval_table.tex
Missing inputs become \\TODO{...} cells so the report compiles even before runs finish.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUTO = Path(__file__).resolve().parent / "_auto"
AUTO.mkdir(parents=True, exist_ok=True)


def _r(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def todo(s="?"): return f"\\TODO{{{s}}}"


def fmt(x, nd=3):
    if x is None: return todo()
    if isinstance(x, float): return f"{x:.{nd}f}"
    return str(x)


def write(name: str, body: str):
    (AUTO / name).write_text(body)
    print(f"[auto] wrote report/_auto/{name}")


def data_stats():
    s = _r(ROOT / "data" / "raw" / "stats.json")
    m = _r(ROOT / "data" / "bin" / "meta.json")
    if not s or not m:
        write("data_stats.tex", todo("dataset statistics not yet available"))
        return
    rows = [
        ("Raw SVGs ingested", s["n_raw"]),
        ("Kept after cleaning", s["n_kept"]),
        ("Dropped: parse fail", s["n_parse_fail"]),
        ("Dropped: too short", s["n_too_short"]),
        ("Dropped: too long", s["n_too_long"]),
        ("Train / Val / Test files", f"{s['n_train']} / {s['n_val']} / {s['n_test']}"),
        ("Vocab size (BPE)", m["vocab_size"]),
        ("Train / Val / Test tokens",
         f"{m['train']['n_tokens']:,} / {m['val']['n_tokens']:,} / {m['test']['n_tokens']:,}"),
        ("Char-len p50 / p90 / p99",
         f"{s['char_len_quantiles']['p50']} / {s['char_len_quantiles']['p90']} / {s['char_len_quantiles']['p99']}"),
    ]
    body = "\\begin{tabular}{lr}\\toprule\nField & Value \\\\\\midrule\n"
    body += "".join(f"{k} & {v} \\\\\n" for k, v in rows)
    body += "\\bottomrule\\end{tabular}\n"
    write("data_stats.tex", body)


def arch_and_results():
    runs = sorted((ROOT / "results" / "runs").glob("*/metrics.json"))
    if not runs:
        write("arch_table.tex", todo("no runs yet"))
        write("results_table.tex", todo("no runs yet"))
        return
    rows_arch, rows_res = [], []
    for p in runs:
        m = json.load(open(p))
        c = m["config"]
        rows_arch.append((m["tag"], c["d_model"], c["n_layers"], c["n_heads"], c["d_ff"], f"{m['n_params']:,}"))
        rows_res.append((m["tag"], m["mup"], fmt(m["lr"], 4), fmt(m["final_val_loss"], 4),
                         fmt(m["wall_clock_s"], 0), fmt(m["tokens_per_s"], 0), fmt(m["gpu_mem_gb"], 2)))
    a = "\\begin{tabular}{lrrrrr}\\toprule\nTag & d\\_model & n\\_layers & n\\_heads & d\\_ff & Params \\\\\\midrule\n"
    a += "".join(" & ".join(map(str, r)) + " \\\\\n" for r in rows_arch)
    a += "\\bottomrule\\end{tabular}\n"
    write("arch_table.tex", a)
    b = "\\begin{tabular}{lrrrrrr}\\toprule\nTag & µP & LR & Val loss & Wall(s) & Tok/s & GPU(GB) \\\\\\midrule\n"
    b += "".join(" & ".join(map(str, r)) + " \\\\\n" for r in rows_res)
    b += "\\bottomrule\\end{tabular}\n"
    write("results_table.tex", b)


def sweeps():
    for kind in ("sp", "mup"):
        s = _r(ROOT / "results" / f"sweep_{kind}.json")
        if not s:
            write(f"sweep_{kind}.tex", todo(f"{kind.upper()} sweep not run"))
            continue
        body = "\\begin{tabular}{rr}\\toprule\nLR & Val loss \\\\\\midrule\n"
        for r in s["results"]:
            body += f"{r['lr']:.0e} & {fmt(r['val_loss'], 4)} \\\\\n"
        body += "\\bottomrule\\end{tabular}\n"
        if s.get("best"):
            body += f"\n\\textbf{{Best LR: {s['best']['lr']:.0e}}} (val loss {s['best']['val_loss']:.4f}).\n"
        write(f"sweep_{kind}.tex", body)


def scaling():
    s = _r(ROOT / "results" / "scaling.json")
    if not s:
        write("scaling_summary.tex", todo("scaling fit not run"))
        return
    lines = []
    for label, v in s.items():
        ext = v["extrapolation"]
        lines.append(
            f"{label}: $\\alpha={v['alpha']:.3f}$, $a={v['a']:.3f}$, $c={v['c']:.3f}$. "
            f"Extrapolated loss at $N={ext['N']:.2e}$ ($10\\times$ XL): "
            f"{ext['median']:.3f} (95\\% CI [{ext['ci_low']:.3f}, {ext['ci_high']:.3f}]).\\\\"
        )
    write("scaling_summary.tex", "\n".join(lines))


def evald():
    e = _r(ROOT / "results" / "eval.json")
    if not e:
        write("eval_table.tex", todo("eval not run"))
        return
    rows = [
        ("Test perplexity", fmt(e.get("test_perplexity"), 3)),
        ("XML validity", fmt(e.get("xml_validity_rate"), 3)),
        ("Render rate", fmt(e.get("render_rate"), 3)),
        ("Structural validity", fmt(e.get("structural_validity_rate"), 3)),
        ("N samples", e.get("n_samples", "--")),
    ]
    body = "\\begin{tabular}{lr}\\toprule\nMetric & Value \\\\\\midrule\n"
    body += "".join(f"{k} & {v} \\\\\n" for k, v in rows)
    body += "\\bottomrule\\end{tabular}\n"
    write("eval_table.tex", body)


if __name__ == "__main__":
    data_stats()
    arch_and_results()
    sweeps()
    scaling()
    evald()
