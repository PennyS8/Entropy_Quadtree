"""
reproduce.py

Validate that the current environment can reproduce a prior results, and
optionally re-run classify.py with the exact parameters recorded in
sidecar JSON or model bundle.

Part of the Quadtree Complexity Analysis for Image Forensics pipeline.

Usage:
    # Validate environment against a feature CSV sidecar
    python3 src/reproduce.py results/features/FFHQ_shannon.csv.json
    
    # Validate environment against a saved model bundle
    python3 src/reproduce.py results/models/stylegan_v1_shannon.joblib
    
    # Validate and re-run classification that produced a report
    python3 src/reproduce.py results/classify/within/stylegan_v1_shannon.txt
    
    # Check all sidecars in results/features/
    python3 src/reproduce.py --all

Output:
    Console report showing environment match/mismatch per dependency.
    Exit code 0 if all checks pass, 1 if any mismatch is found.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import setup_logging, get_logger

log = get_logger(__name__)

# Minimum required versions

# Versions pinned in requirements.txt: checks warn if environment differs.
REQUIRED_VERSIONS = {
    "Pillow":       "10.0",
    "numpy":        "1.24",
    "scikit-learn": "1.3",
    "joblib":       "1.3",
    "requests":     "2.31",
    "matplotlib":   "3.7"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate environment reproducibility for the QIA pipeline."
    )
    parser.add_argument("target", nargs="?", default=None,
                        help="Path to a sidecar .json, .joblib bundle, or"
                        "classify report .txt to validate against.")
    parser.add_argument("--all", action="store_true",
                        help="Check all sidecar JSONs in results/features/")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run classify.py using parameters from a "
                        "model bundle (requires --target pointing to a .joblib).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    return parser.parse_args()


# Environment snapshot

def get_env_snapshot() -> dict:
    """Return current Python version and installed package versions."""
    versions = {}
    for pkg in REQUIRED_VERSIONS:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "NOT INSTALLED"
    return {
        "python":   platform.python_version(),
        "platform": platform.platform(),
        "packages": versions
    }


def check_version(pkg: str, installed: str, required: str) -> tuple[bool, str]:
    """Return (ok, message). Passes if installed >= required (semver major.minor)."""
    if installed == "NOT INSTALLED":
        return False, f"  x  {pkg:<20} NOT INSTALLED  (requires >= {required})"
    try:
        inst_parts  = tuple(int(x) for x in installed.split(".")[:2])
        req_parts   = tuple(int(x) for x in required.split(".")[:2])
        ok = inst_parts >= req_parts
        symbol  = "v" if ok else "x"
        note    = "" if ok else f"  <- requires >= {required}"
        return ok, f"  {symbol}  {pkg:<20} {installed}{note}"
    except ValueError:
        return True, f"  ?  {pkg:<20} {installed}  (could not compare)"


# Sidecar JSON validation

def validate_sidecar(path: str, env: dict) -> bool:
    """Validate a feature CSV sidecar JSON against the current environment."""
    if not os.path.exitsts(path):
        log.error("Sidecar not found: %s", path)
        return False
    
    with open(path) as f:
        meta = json.load(f)
    
    print(f"\nSidecar: {path}")
    print(f"  Dataset:    {meta.get('dataset', '—')}")
    print(f"  Name:       {meta.get('name', '—')}")
    print(f"  Method:     {meta.get('method', '—')}  leaf_size={meta.get('leaf_size', '—')}")
    print(f"  Resize:     {meta.get('resize', '—')}px")
    print(f"  Threshold:  {meta.get('threshold', 'off')}")
    print(f"  Timestamp:  {meta.get('timestamp', '—')}")
    print(f"  Total:      {meta.get('total', '—')} images  ({meta.get('errors', 0)} errors)")
    return True


# Bundle validation

def validate_bundle(path: str, env: dict) -> bool:
    """Validate a .joblib model bundle against the current environment."""
    try:
        import joblib
        bundle = joblib.load(path)
    except Exception as e:
        log.error("Could not load bundle %s: %s", path, e)
        return False
 
    print(f"\nModel bundle: {path}")
    print(f"  Classifier: {bundle.get('clf_name', '—')}")
    print(f"  Method:     {bundle.get('method', '—')}  leaf_size={bundle.get('leaf_size', '—')}")
    print(f"  Resize:     {bundle.get('resize', '—')}px")
    print(f"  Seed:       {bundle.get('seed', '—')}")
    print(f"  Accuracy:   {bundle.get('accuracy', '—')}")
    print(f"  Trained on: {bundle.get('trained_on', '—')}")
    print(f"  Saved at:   {bundle.get('saved_at', '—')}")
 
    n_feat = len(bundle.get('feature_cols', []))
    tg = sum(1 for f in bundle.get('feature_cols', []) if f.startswith('tree_grid_'))
    print(f"  Features:   {n_feat}  ({n_feat - tg} scalar + {tg} spatial cells)")
    return True


# Report validation

def validate_report(path: str) -> bool:
    """Parse a classify.py report .txt and display its reproducibility haeder."""
    if not os.path.exists(path):
        log.error("Report not found: %s", path)
        return False
    
    with open(path) as f:
        lines = f.readlines()
    
    print(f"\nReport: {path}")
    # Print the header section (everything before the first classifier result)
    for line in lines[:20]:
        stripped = line.rstrip()
        if stripped:
            print(f"  {stripped}")
        if line.startswith("="):
            break
    return True


# Main

def main():
    args = parse_args()
    setup_logging(args.verbose)
    
    env = get_env_snapshot()
    
    # Environment check
    print("── Environment ──────────────────────────────────────────")
    print(f"  Python:   {env['python']}")
    print(f"  Platform: {env['platform']}")
    print()
    
    all_ok = True
    for pkg, required in REQUIRED_VERSIONS.items():
        installed = env["packages"].get(pkg, "NOT INSTALLED")
        ok, msg = check_version(pkg, installed, required)
        print(msg)
        if not ok:
            all_ok = False
 
    print()
    if all_ok:
        print("  ✓  All dependencies meet minimum version requirements.")
    else:
        print("  ✗  Some dependencies are out of date.")
        print("     Run: pip install -r requirements.txt")
    
    # Target validation
    targets = []
    
    if args.all:
        feat_dir = config.DIRS["features"]
        if os.path.isdir(feat_dir):
            targets = sorted(
                os.path.join(feat_dir, f)
                for f in os.listdir(feat_dir)
                if f.endswith(".json")
            )
            if not targets:
                log.warning("No sidecar JSONs found in %s", feat_dir)
        else:
            log.warning("Features directory not found: %s", feat_dir)
    
    elif args.target:
        targets = [args.target]
    
    for target in targets:
        print("\n" + "─" * 56)
        if target.endswith(".json"):
            validate_sidecar(target, env)
        elif target.endswith(".joblib"):
            validate_bundle(target, env)
        elif target.endswith(".txt"):
            validate_report(target)
        else:
            log.warning("Unrecognised file type: %s", target)
    
    
    # Re-run
    if args.rerun:
        if not args.target or not args.target.endswith(".joblib"):
            log.error("--rerun requires --target pointing to a .joblib bundle")
            sys.exit(1)
        try:
            import joblib as jl
            bundle = jl.load(args.target)
        except Exception as e:
            log.error("Could not load bundle: %s", e)
            sys.exit(1)
        
        trained_on = bundle.get("trained_on", "")
        method     = bundle.get("method", "")
        leaf_size  = bundle.get("leaf_size", "")
        resize     = bundle.get("resize", "")
        seed       = bundle.get("seed", 42)
        name       = os.path.splitext(os.path.basename(args.target))[0]
        
        csv_paths = [p.strip() for p in trained_on.split(",") if p.strip()]
        if not csv_paths:
            log.error("Bundle 'trained_on' is empty — cannot reconstruct command.")
            sys.exit(1)
        
        cmd_parts = ["python3 src/classify.py"] + csv_paths + [
            "--fast --balance --prune-features 0.0",
            f"--method {method}",
            f"--leaf-size {leaf_size}",
            f"--resize {resize}" if resize else "",
            f"--seed {seed}",
            f"--save-model {name}"
        ]
        cmd = " \\\n    ".join(p for p in cmd_parts if p)
        
        print("\n── Re-run command ───────────────────────────────────────")
        print(cmd)
        print()
        
        response = input("Run this command now? [y/N] ").strip().lower()
        if response == "y":
            import subprocess
            full_cmd = " ".join(p for p in cmd_parts if p).replace(" \\\n    ", " ")
            result = subprocess.run(full_cmd, shell=True)
            sys.exit(result.returncode)
    
    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()