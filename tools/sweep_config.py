#!/usr/bin/env python
"""Turn a sweep config in `benchmarks/` into the command line the benchmark takes.

`docs/INFERENCE.md` documents a config schema and states plainly that
`scripts/run_inference_benchmark.py` is driven by flags and does not read YAML,
so the configs are translated to flags by hand or by a Makefile target. Doing
that by hand is how a config and the command that is supposed to implement it
drift apart, and a config that no longer matches the run it claims to describe
is worse than no config, because it is a record that lies.

This is the translator. It reads a config, validates the same required keys the
CI job validates, and prints the flag string. The shell driver in
`scripts/sweep.sh` runs the benchmark through it, so the config is the single
source of truth for what was run.

It also reports what it could not translate, on stderr, rather than quietly
dropping it. Two parts of the schema have no flag today.

`matrix.runtimes` has no `--runtimes` flag. The benchmark probes every backend
it knows about and records the ones that could not be built in
`unavailable_backends`, so the runtime list in a config is a description of what
is expected rather than a selection.

`exclude` has no flag either. Exclusion rules drop individual cells from the
cartesian product and a flag list cannot express that, so the excluded cells are
printed as a warning and the sweep runs the full product.

Both of those are real gaps between the documented schema and the shipped
script, and the honest thing is to print them every time rather than to leave
the reader believing a config key had an effect it did not have.

Usage.

    python tools/sweep_config.py cpu_only
    python tools/sweep_config.py benchmarks/gpu_full.yaml --print flags
    python tools/sweep_config.py cpu_only --print summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

BENCHMARKS_DIR = os.path.join(_REPO_ROOT, "benchmarks")

# The keys CI already checks for. Kept identical on purpose so a config that
# passes here passes there.
REQUIRED_KEYS = ("name", "description", "hardware", "data", "run", "matrix")
MATRIX_KEYS = ("models", "runtimes", "precisions", "batch_sizes")
SINGULAR = {
    "models": "model",
    "runtimes": "runtime",
    "precisions": "precision",
    "batch_sizes": "batch_size",
}

# Config key to flag. Only the keys the benchmark script actually accepts are
# here. Anything in the schema and not in this table is reported as untranslated
# rather than silently ignored.
SCALAR_FLAGS = (
    ("data", "data_path", "--data-path"),
    ("data", "sample_size", "--sample-size"),
    ("run", "checkpoint", "--checkpoint"),
    ("run", "output", "--output"),
    ("run", "repeats", "--repeats"),
    ("run", "warmup", "--warmup"),
    ("run", "batch_size", "--batch-size"),
    ("run", "timing_batches", "--timing-batches"),
    ("run", "engine_dir", "--engine-dir"),
)

# Matrix keys that do map to a flag. These take space separated values rather
# than a comma separated string, because that is what the argparse nargs plus
# definition in the benchmark script accepts.
LIST_FLAGS = (
    ("models", "--models"),
    ("precisions", "--precisions"),
    ("batch_sizes", "--batch-sizes"),
)

UNTRANSLATED_MATRIX_KEYS = {
    "runtimes": (
        "the benchmark has no --runtimes flag. It probes every backend it knows "
        "about and records the ones it could not build under "
        "unavailable_backends, so this list is a statement of what is expected "
        "rather than a selection"
    ),
}


class ConfigError(Exception):
    """Raised when a config cannot be read or does not satisfy the schema."""


def config_path(name_or_path: str) -> str:
    """Resolve a config name or a path to a file on disk.

    A bare name is looked up under `benchmarks/`, so `cpu_only` and
    `benchmarks/cpu_only.yaml` both work and a caller does not have to know
    where the configs live.
    """
    candidates = [name_or_path]
    if not name_or_path.endswith((".yaml", ".yml")):
        candidates.append(os.path.join(BENCHMARKS_DIR, f"{name_or_path}.yaml"))
        candidates.append(os.path.join(BENCHMARKS_DIR, f"{name_or_path}.yml"))
    else:
        candidates.append(os.path.join(BENCHMARKS_DIR, os.path.basename(name_or_path)))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    available = ", ".join(sorted(known_configs())) or "none found"
    raise ConfigError(
        f"no sweep config named {name_or_path}. Available configs are {available}"
    )


def known_configs() -> List[str]:
    """List the config names available under benchmarks/."""
    if not os.path.isdir(BENCHMARKS_DIR):
        return []
    names = []
    for entry in os.listdir(BENCHMARKS_DIR):
        if entry.endswith((".yaml", ".yml")):
            names.append(os.path.splitext(entry)[0])
    return names


def load_config(name_or_path: str) -> Tuple[Dict[str, Any], str]:
    """Read and validate a sweep config, returning the config and its path."""
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "pyyaml is required to read a sweep config and is not installed. "
            "It is deliberately not in requirements.txt because nothing in the "
            "library imports it. Install it with pip install pyyaml"
        ) from exc

    path = config_path(name_or_path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"could not read {path}. {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid yaml. {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"{path} does not contain a mapping at the top level")

    validate(config, path)
    return config, path


def validate(config: Dict[str, Any], path: str) -> None:
    """Check a config against the schema documented in docs/INFERENCE.md."""
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ConfigError(f"{path} is missing the required keys {missing}")

    matrix = config.get("matrix") or {}
    for key in MATRIX_KEYS:
        value = matrix.get(key)
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{path} matrix.{key} must be a non empty list")

    allowed = set(SINGULAR.values())
    for rule in config.get("exclude") or []:
        if not isinstance(rule, dict):
            raise ConfigError(f"{path} has an exclude rule that is not a mapping")
        unknown = set(rule) - allowed
        if unknown:
            raise ConfigError(
                f"{path} exclude rule names unknown dimensions {sorted(unknown)}. "
                f"A typo here silently excludes nothing and the sweep quietly grows"
            )


def batch_sizes(config: Dict[str, Any]) -> List[int]:
    """Return the batch sizes the sweep should cover.

    `run.batch_size` is the single run default and `matrix.batch_sizes` is the
    sweep. When both are present the sweep wins, which is the rule stated in the
    schema section of docs/INFERENCE.md.
    """
    matrix = config.get("matrix") or {}
    sizes = matrix.get("batch_sizes")
    if sizes:
        return [int(value) for value in sizes]
    fallback = (config.get("run") or {}).get("batch_size")
    return [int(fallback)] if fallback else []


def to_flags(config: Dict[str, Any]) -> List[str]:
    """Translate a config into the argument list the benchmark script accepts."""
    flags: List[str] = []

    for section, key, flag in SCALAR_FLAGS:
        value = (config.get(section) or {}).get(key)
        if value is None or value == "":
            continue
        flags.extend([flag, str(value)])

    if (config.get("data") or {}).get("synthetic"):
        # A store true flag, so it is passed only when the config asks for it.
        flags.append("--synthetic")

    matrix = config.get("matrix") or {}
    for key, flag in LIST_FLAGS:
        values = matrix.get(key)
        if not values:
            continue
        flags.append(flag)
        flags.extend(str(value) for value in values)

    return flags


def drop_flags(flags: List[str], omit: Sequence[str]) -> List[str]:
    """Remove named flags, and the values that follow them, from a flag list.

    The caller that needs this is the sweep driver, which takes the config's
    output directory out of the translated command so it can substitute its own
    timestamped run directory. Passing both and relying on argparse to keep the
    last one would work and would print a command with the same flag twice,
    which is the kind of thing that makes a reader stop trusting the log.
    """
    if not omit:
        return list(flags)
    wanted = set(omit)
    out: List[str] = []
    index = 0
    while index < len(flags):
        token = flags[index]
        if token in wanted:
            index += 1
            # Skip the values that belong to the dropped flag. Everything up to
            # the next token that starts with a dash is one of its values, which
            # covers both the single value flags and the nargs plus ones.
            while index < len(flags) and not flags[index].startswith("--"):
                index += 1
            continue
        out.append(token)
        index += 1
    return out


def untranslated(config: Dict[str, Any]) -> List[str]:
    """List the parts of the config that have no flag, with the reason for each."""
    notes: List[str] = []
    matrix = config.get("matrix") or {}
    for key, reason in UNTRANSLATED_MATRIX_KEYS.items():
        values = matrix.get(key)
        if values:
            listed = ", ".join(str(value) for value in values)
            notes.append(f"matrix.{key} [{listed}] was not translated, because {reason}")

    rules = config.get("exclude") or []
    if rules:
        rendered = ", ".join(
            "{" + ", ".join(f"{k}={v}" for k, v in sorted(rule.items())) + "}"
            for rule in rules
        )
        notes.append(
            f"{len(rules)} exclude rule(s) {rendered} were not translated, because "
            "a flag list cannot drop individual cells from a cartesian product. "
            "The sweep runs the full product and the excluded cells, if they run "
            "at all, appear in the results"
        )

    if (config.get("calibration") or {}) and "int8" in (config.get("matrix") or {}).get(
        "precisions", []
    ):
        notes.append(
            "the calibration block was not translated. The int8 calibration set "
            "is configured where the engines are built, in "
            "scripts/build_trt_engines.py, and not where the sweep is run"
        )

    return notes


def cell_count(config: Dict[str, Any]) -> int:
    """Count the cells in the cartesian product after the exclusion rules apply."""
    matrix = config.get("matrix") or {}
    rules = config.get("exclude") or []
    total = 0
    for model in matrix.get("models", []):
        for runtime in matrix.get("runtimes", []):
            for precision in matrix.get("precisions", []):
                for batch in batch_sizes(config):
                    cell = {
                        "model": model,
                        "runtime": runtime,
                        "precision": precision,
                        "batch_size": batch,
                    }
                    if any(
                        all(cell.get(k) == v for k, v in rule.items()) for rule in rules
                    ):
                        continue
                    total += 1
    return total


def summary(config: Dict[str, Any], path: str) -> str:
    """Render a short human readable description of what a config will run."""
    matrix = config.get("matrix") or {}
    lines = [
        f"config     {path}",
        f"name       {config.get('name')}",
        f"hardware   {config.get('hardware')}",
        f"output     {(config.get('run') or {}).get('output')}",
        f"models     {', '.join(str(v) for v in matrix.get('models', []))}",
        f"runtimes   {', '.join(str(v) for v in matrix.get('runtimes', []))}",
        f"precisions {', '.join(str(v) for v in matrix.get('precisions', []))}",
        f"batches    {', '.join(str(v) for v in batch_sizes(config))}",
        f"cells      {cell_count(config)} after {len(config.get('exclude') or [])} exclusions",
    ]
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sweep_config.py",
        description=(
            "Translate a sweep config in benchmarks/ into the command line "
            "scripts/run_inference_benchmark.py accepts."
        ),
    )
    parser.add_argument("config", help="Config name such as cpu_only, or a path to a yaml file.")
    parser.add_argument(
        "--print",
        dest="what",
        default="flags",
        choices=("flags", "name", "output", "hardware", "summary", "json"),
        help="What to print on stdout. Default %(default)s",
    )
    parser.add_argument(
        "--omit",
        action="append",
        default=[],
        metavar="FLAG",
        help=(
            "Drop this flag and its values from the translated command. "
            "Repeatable. The sweep driver uses it to replace the config's "
            "--output with its own run directory."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the untranslated key warnings on stderr.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        config, path = load_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not args.quiet:
        for note in untranslated(config):
            print(f"warning. {note}", file=sys.stderr)

    if args.what == "flags":
        print(" ".join(drop_flags(to_flags(config), args.omit)))
    elif args.what == "name":
        print(config.get("name", os.path.splitext(os.path.basename(path))[0]))
    elif args.what == "output":
        print((config.get("run") or {}).get("output", "results"))
    elif args.what == "hardware":
        print(config.get("hardware", "unknown"))
    elif args.what == "summary":
        print(summary(config, path))
    elif args.what == "json":
        print(
            json.dumps(
                {
                    "path": path,
                    "name": config.get("name"),
                    "hardware": config.get("hardware"),
                    "output": (config.get("run") or {}).get("output"),
                    "flags": drop_flags(to_flags(config), args.omit),
                    "cells": cell_count(config),
                    "untranslated": untranslated(config),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
