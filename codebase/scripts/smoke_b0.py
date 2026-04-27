#!/usr/bin/env python3
"""End-to-end B0 (no-training) smoke test on a tiny EFCAMDAT sample.

Produces a self-contained, timestamped output directory named
``sample-benchmark-{model_hash}-{model_name}-{timestamp}/`` containing:

* ``paths.yaml``              — the path config used for this run
* ``sample/*.csv``            — the sampled slices of each corpus
* ``last/``                   — B0 checkpoint + all eval artefacts
* ``aggregate.{json,csv}``    — headline numbers

Usage::

    py scripts/smoke_b0.py --n_efcamdat 100 --n_transfer 50

``py`` is a pyenv alias for Python 3.10. Works on CPU; no GPU required.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

CODEBASE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CODEBASE_ROOT.parent
SRC = CODEBASE_ROOT / "src"


def _shell_env() -> dict:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC}:{pythonpath}" if pythonpath else str(SRC)
    # Keep HF offline behaviour permissive for first-time runs.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=_shell_env(), check=True)


def _sample_stratified(df: pd.DataFrame, n: int, key: str, seed: int) -> pd.DataFrame:
    """Take n rows, evenly across unique ``key`` values where possible."""
    if key not in df.columns or df[key].isna().all():
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    groups = df.dropna(subset=[key]).groupby(key, group_keys=False)
    per_group = max(n // max(groups.ngroups, 1), 1)
    sampled = groups.apply(lambda g: g.sample(n=min(per_group, len(g)), random_state=seed))
    if len(sampled) < n:
        extra = df.drop(sampled.index, errors="ignore").sample(
            n=min(n - len(sampled), len(df) - len(sampled)),
            random_state=seed,
        )
        sampled = pd.concat([sampled, extra])
    return sampled.head(n).reset_index(drop=True)


def _model_identity(model_name: str) -> tuple[str, str]:
    """Return (short_hash, model_name) for the given HF model id.

    Prefers the HF Hub commit sha; falls back to a config-content hash.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_name)
        if info.sha:
            return info.sha[:8], model_name
    except Exception as e:
        print(f"[warn] could not fetch HF sha for {model_name}: {e}")

    # Fallback: hash the public AutoConfig dict.
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name).to_dict()
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    return digest[:8], model_name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_efcamdat", type=int, default=100)
    p.add_argument("--n_transfer", type=int, default=50)
    p.add_argument("--model_name", default="gpt2")
    p.add_argument(
        "--efcamdat_source",
        type=Path,
        default=Path(
            "./data/splits/"
            "norm-EFCAMDAT-test.csv"
        ),
    )
    p.add_argument(
        "--andrew100k_source",
        type=Path,
        default=Path(
            "./data/splits/"
            "norm-andrew100k-remainder.csv"
        ),
    )
    p.add_argument(
        "--celva_source",
        type=Path,
        default=Path(
            "./data/splits/"
            "norm-CELVA-SP.csv"
        ),
    )
    p.add_argument(
        "--kupa_source",
        type=Path,
        default=Path(
            "./data/splits/"
            "norm-KUPA-KEYS.csv"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_root",
        type=Path,
        default=CODEBASE_ROOT,
        help="Parent directory that will contain sample-benchmark-*/",
    )
    p.add_argument("--skip_cloze", action="store_true")
    args = p.parse_args()

    print(f"[smoke] identifying model {args.model_name} ...")
    model_hash, model_name = _model_identity(args.model_name)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"sample-benchmark-{model_hash}-{model_name}-{timestamp}"
    run_dir = args.out_root / run_name
    sample_dir = run_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] run_dir: {run_dir}")

    # 1. Sample each corpus.
    print(f"[smoke] sampling {args.n_efcamdat} rows from EFCAMDAT, "
          f"{args.n_transfer} from each transfer corpus (seed={args.seed})")
    plans = [
        ("efcamdat_test", args.efcamdat_source, args.n_efcamdat, "cefr_level"),
        ("andrew100k_remainder", args.andrew100k_source, args.n_transfer, "cefr_label"),
        ("celva_sp", args.celva_source, args.n_transfer, "cefr_level"),
        ("kupa_keys", args.kupa_source, args.n_transfer, "cefr_level"),
    ]
    sample_files: dict[str, Path] = {}
    for name, src, n, strat_key in plans:
        df = pd.read_csv(src, low_memory=False)
        sampled = _sample_stratified(df, n=n, key=strat_key, seed=args.seed)
        out = sample_dir / f"{name}.csv"
        sampled.to_csv(out, index=False)
        sample_files[name] = out
        print(f"  {name}: {len(sampled)} rows -> {out.name}")

    # 2. Write a scoped paths.yaml.
    paths_yaml = run_dir / "paths.yaml"
    paths_yaml.write_text(
        "data_root: {root}\n"
        "efcamdat:\n"
        "  train: {efcamdat}\n"
        "  remainder: {efcamdat}\n"
        "  test: {efcamdat}\n"
        "transfer:\n"
        "  andrew100k:\n"
        "    remainder: {andrew}\n"
        "    test_label: {andrew}\n"
        "  celva_sp: {celva}\n"
        "  kupa_keys: {kupa}\n"
        "cache_root: {cache}\n"
        "runs_root: {runs}\n".format(
            root=sample_dir,
            efcamdat=sample_files["efcamdat_test"].name,
            andrew=sample_files["andrew100k_remainder"].name,
            celva=sample_files["celva_sp"].name,
            kupa=sample_files["kupa_keys"].name,
            cache=run_dir / "cache",
            runs=run_dir,
        )
    )
    print(f"[smoke] wrote {paths_yaml}")

    # 3. Materialise B0 checkpoint into run_dir/last/.
    py = sys.executable
    _run(
        [
            py, "scripts/03_prepare_b0_checkpoint.py",
            "--config", "configs/b0_native.yaml",
            "--paths_config", str(paths_yaml),
            "--run_dir", str(run_dir),
        ],
        cwd=CODEBASE_ROOT,
    )

    checkpoint = run_dir / "last"

    # 4. Stratified PPL on sampled EFCAMDAT + sampled transfer corpora.
    _run(
        [
            py, "scripts/20_eval_ppl.py",
            "--checkpoint", str(checkpoint),
            "--split", "all",
            "--paths_config", str(paths_yaml),
            "--batch_size", "4",
        ],
        cwd=CODEBASE_ROOT,
    )

    # 5. Cloze accuracy (optional).
    if not args.skip_cloze:
        _run(
            [
                py, "scripts/21_eval_cloze.py",
                "--checkpoint", str(checkpoint),
                "--paths_config", str(paths_yaml),
                "--max_examples", str(min(args.n_efcamdat, 50)),
            ],
            cwd=CODEBASE_ROOT,
        )

    # 6. Aggregate only this run's artefacts (avoids picking up stale
    # sibling sample-benchmark-* dirs if you run smoke_b0.py multiple
    # times).
    summary: dict = {"run": run_name}
    for fname in ("ppl_in_domain.json", "ppl_transfer.json", "cloze.json"):
        fpath = checkpoint / fname
        if fpath.exists():
            summary[fname.replace(".json", "")] = json.loads(fpath.read_text())
    (run_dir / "aggregate.json").write_text(json.dumps(summary, indent=2))

    # Flat headline numbers.
    flat: dict = {"run": run_name, "model": f"{model_name}@{model_hash}"}
    if "ppl_in_domain" in summary and "efcamdat_test" in summary["ppl_in_domain"]:
        flat["ppl_indomain_overall"] = summary["ppl_in_domain"]["efcamdat_test"]["overall"]
    if "ppl_transfer" in summary:
        for corpus_name, report in summary["ppl_transfer"].items():
            flat[f"ppl_{corpus_name}"] = report["overall"]
    if "cloze" in summary:
        flat["cloze_top1"] = summary["cloze"].get("top1")
        flat["cloze_top5"] = summary["cloze"].get("top5")
    pd.DataFrame([flat]).to_csv(run_dir / "aggregate.csv", index=False)

    print("\n============== RESULTS ==============")
    for k, v in flat.items():
        print(f"  {k}: {v}")
    print("=====================================")
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
