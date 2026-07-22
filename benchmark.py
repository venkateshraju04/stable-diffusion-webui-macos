#!/usr/bin/env python3
"""
Benchmark script for Stable Diffusion WebUI macOS.

Captures performance and memory metrics for before/after comparison.

Usage:
    python benchmark.py --label before    # Run benchmark, save as "before"
    python benchmark.py --label after     # Run benchmark, save as "after"
    python benchmark.py --compare before after  # Generate comparison report
"""

import argparse
import gc
import json
import os
import platform
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

# Add the repo root to path so we can import modules
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

RESULTS_DIR = SCRIPT_DIR / "benchmark_results"


def get_system_info():
    """Collect system information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    try:
        import subprocess
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True
        ).strip()
        info["chip"] = chip
    except Exception:
        info["chip"] = "unknown"

    try:
        import subprocess
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True
        ).strip())
        info["total_memory_gb"] = round(mem_bytes / (1024 ** 3), 1)
    except Exception:
        info["total_memory_gb"] = "unknown"

    return info


def setup_sys_path():
    """Add repositories to sys.path like launch.py does."""
    repos_dir = SCRIPT_DIR / "repositories"
    if repos_dir.exists():
        for repo in repos_dir.iterdir():
            if repo.is_dir():
                sys.path.insert(0, str(repo))
        
        # specifically for stable-diffusion which has an ldm module
        sd_repo = repos_dir / "stable-diffusion-stability-ai"
        if not sd_repo.exists():
            sd_repo = repos_dir / "generative-models" 
        
        if sd_repo.exists():
            sys.path.insert(0, str(sd_repo))

def get_torch_info():
    """Collect PyTorch and MPS information."""
    import torch
    info = {
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }
    return info


def bytes_to_gb(b):
    """Convert bytes to GB with 2 decimal places."""
    return round(b / (1024 ** 3), 2)


class MPSMemoryTracker:
    """Track MPS memory usage in a background thread."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_allocated = 0
        self.peak_driver = 0
        self.samples = []
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._track, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _track(self):
        import torch
        while self._running:
            try:
                allocated = torch.mps.current_allocated_memory()
                driver = torch.mps.driver_allocated_memory()
                self.peak_allocated = max(self.peak_allocated, allocated)
                self.peak_driver = max(self.peak_driver, driver)
                self.samples.append({
                    "time": time.time(),
                    "allocated_gb": bytes_to_gb(allocated),
                    "driver_gb": bytes_to_gb(driver),
                })
            except Exception:
                pass
            time.sleep(self.interval)

    def get_stats(self):
        return {
            "peak_allocated_gb": bytes_to_gb(self.peak_allocated),
            "peak_driver_gb": bytes_to_gb(self.peak_driver),
            "num_samples": len(self.samples),
        }


def get_system_memory():
    """Get current system memory usage via psutil."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": bytes_to_gb(mem.total),
            "available_gb": bytes_to_gb(mem.available),
            "used_gb": bytes_to_gb(mem.used),
            "percent": mem.percent,
        }
    except ImportError:
        return {"error": "psutil not installed"}


def get_mps_memory():
    """Get current MPS memory stats."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return {
                "allocated_gb": bytes_to_gb(torch.mps.current_allocated_memory()),
                "driver_gb": bytes_to_gb(torch.mps.driver_allocated_memory()),
            }
    except Exception:
        pass
    return {"allocated_gb": 0, "driver_gb": 0}


def run_benchmark(label):
    """Run the full benchmark suite."""
    setup_sys_path()
    import torch

    print(f"\n{'='*60}")
    print(f"  Stable Diffusion WebUI macOS — Benchmark")
    print(f"  Label: {label}")
    print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    results = {
        "label": label,
        "system": get_system_info(),
        "torch": get_torch_info(),
        "metrics": {},
    }

    # --- Metric 1: System memory at rest ---
    print("[1/6] Capturing idle system memory...")
    results["metrics"]["system_memory_idle"] = get_system_memory()
    results["metrics"]["mps_memory_idle"] = get_mps_memory()

    # --- Metric 2: Startup / import time ---
    print("[2/6] Measuring import time...")
    t0 = time.time()
    try:
        from modules import shared, sd_models, devices
        from modules.processing import StableDiffusionProcessingTxt2Img, process_images
        import_time = time.time() - t0
        results["metrics"]["import_time_s"] = round(import_time, 2)
        print(f"       Import time: {import_time:.2f}s")
    except Exception as e:
        results["metrics"]["import_time_s"] = -1
        results["metrics"]["import_error"] = str(e)
        print(f"       Import failed: {e}")
        print("       Cannot proceed without imports. Saving partial results.")
        save_results(results, label)
        return results

    # --- Metric 3: Model load time + idle memory ---
    print("[3/6] Loading model...")
    model_path = SCRIPT_DIR / "models" / "Stable-diffusion"
    safetensor_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.ckpt"))

    if not safetensor_files:
        print("       ERROR: No model files found in models/Stable-diffusion/")
        print("       Please download a model first.")
        results["metrics"]["model_load_error"] = "No model files found"
        save_results(results, label)
        return results

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    t0 = time.time()
    try:
        # Force model load by accessing shared.sd_model
        from modules import initialize_util
        initialize_util.fix_torch_version()
        initialize_util.fix_pytorch_lightning()
        initialize_util.fix_asyncio_event_loop_policy()

        from modules import sd_models
        sd_models.setup_model()
        sd_models.list_models()

        # Load the first available model
        from modules import shared
        if shared.sd_model is None:
            sd_models.load_model()

        model_load_time = time.time() - t0
        results["metrics"]["model_load_time_s"] = round(model_load_time, 2)
        print(f"       Model load time: {model_load_time:.2f}s")
    except Exception as e:
        results["metrics"]["model_load_time_s"] = -1
        results["metrics"]["model_load_error"] = str(e)
        print(f"       Model load failed: {e}")
        save_results(results, label)
        return results

    # Memory after model load
    results["metrics"]["mps_memory_after_load"] = get_mps_memory()
    results["metrics"]["system_memory_after_load"] = get_system_memory()
    print(f"       MPS memory after load: {results['metrics']['mps_memory_after_load']}")

    # --- Metric 4: Generation speed + peak memory ---
    print("[4/6] Running generation benchmark (512x512, 20 steps, Euler)...")
    tracker = MPSMemoryTracker(interval=0.3)

    try:
        from modules.processing import StableDiffusionProcessingTxt2Img, process_images
        from modules import shared

        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt="a photograph of a cat sitting on a windowsill, natural lighting",
            negative_prompt="",
            seed=42,
            sampler_name="Euler",
            steps=20,
            cfg_scale=7.0,
            width=512,
            height=512,
            batch_size=1,
            n_iter=1,
        )

        tracker.start()
        t0 = time.time()
        processed = process_images(p)
        gen_time = time.time() - t0
        tracker.stop()

        results["metrics"]["generation_time_s"] = round(gen_time, 2)
        results["metrics"]["generation_its_per_s"] = round(20 / gen_time, 2)
        results["metrics"]["mps_memory_peak"] = tracker.get_stats()
        print(f"       Generation time: {gen_time:.2f}s ({20/gen_time:.1f} it/s)")
        print(f"       Peak MPS allocated: {tracker.get_stats()['peak_allocated_gb']} GB")
        print(f"       Peak MPS driver: {tracker.get_stats()['peak_driver_gb']} GB")

        # Save the generated image as proof
        if processed and processed.images:
            img_path = RESULTS_DIR / f"{label}_sample.png"
            processed.images[0].save(str(img_path))
            print(f"       Sample image saved: {img_path.name}")

    except Exception as e:
        tracker.stop()
        results["metrics"]["generation_time_s"] = -1
        results["metrics"]["generation_error"] = str(e)
        print(f"       Generation failed: {e}")

    # --- Metric 5: Post-GC memory ---
    print("[5/6] Running garbage collection...")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    time.sleep(1)  # Let the system settle

    results["metrics"]["mps_memory_post_gc"] = get_mps_memory()
    results["metrics"]["system_memory_post_gc"] = get_system_memory()
    print(f"       MPS memory post-GC: {results['metrics']['mps_memory_post_gc']}")

    # --- Metric 6: Summary ---
    print(f"\n[6/6] Summary")
    print(f"       Python:          {results['system']['python_version'].split()[0]}")
    print(f"       PyTorch:         {results['torch']['torch_version']}")
    print(f"       MPS available:   {results['torch']['mps_available']}")
    if results['metrics'].get('generation_time_s', -1) > 0:
        print(f"       Gen time:        {results['metrics']['generation_time_s']}s")
        print(f"       Peak MPS mem:    {results['metrics']['mps_memory_peak']['peak_driver_gb']} GB")
        print(f"       Post-GC MPS mem: {results['metrics']['mps_memory_post_gc']['driver_gb']} GB")

    save_results(results, label)
    return results


def save_results(results, label):
    """Save benchmark results to JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)

    filepath = RESULTS_DIR / f"{label}.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {filepath}")


def compare_results(label_before, label_after):
    """Generate a comparison report between two benchmark runs."""
    before_path = RESULTS_DIR / f"{label_before}.json"
    after_path = RESULTS_DIR / f"{label_after}.json"

    if not before_path.exists():
        print(f"ERROR: {before_path} not found. Run benchmark with --label {label_before} first.")
        return
    if not after_path.exists():
        print(f"ERROR: {after_path} not found. Run benchmark with --label {label_after} first.")
        return

    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    print(f"\n{'='*70}")
    print(f"  Benchmark Comparison: {label_before} vs {label_after}")
    print(f"{'='*70}\n")

    # Build comparison rows
    rows = []

    # Python version
    py_before = before["system"]["python_version"].split()[0]
    py_after = after["system"]["python_version"].split()[0]
    rows.append(("Python version", py_before, py_after,
                  "upgraded ✅" if py_before != py_after else "same"))

    # PyTorch version
    pt_before = before["torch"]["torch_version"]
    pt_after = after["torch"]["torch_version"]
    rows.append(("PyTorch version", pt_before, pt_after,
                  "upgraded ✅" if pt_before != pt_after else "same"))

    # Numeric metrics
    numeric_metrics = [
        ("Import time", "import_time_s", "s", True),  # lower is better
        ("Model load time", "model_load_time_s", "s", True),
        ("Generation time (512×512)", "generation_time_s", "s", True),
        ("Generation speed", "generation_its_per_s", " it/s", False),  # higher is better
    ]

    for name, key, unit, lower_is_better in numeric_metrics:
        val_b = before["metrics"].get(key, -1)
        val_a = after["metrics"].get(key, -1)
        if val_b > 0 and val_a > 0:
            if lower_is_better:
                pct = ((val_b - val_a) / val_b) * 100
                change = f"-{pct:.0f}% ✅" if pct > 0 else f"+{abs(pct):.0f}% ⚠️"
            else:
                pct = ((val_a - val_b) / val_b) * 100
                change = f"+{pct:.0f}% ✅" if pct > 0 else f"-{abs(pct):.0f}% ⚠️"
            rows.append((name, f"{val_b}{unit}", f"{val_a}{unit}", change))
        else:
            rows.append((name, str(val_b), str(val_a), "N/A"))

    # Memory metrics
    mem_metrics = [
        ("Peak MPS memory", lambda r: r["metrics"].get("mps_memory_peak", {}).get("peak_driver_gb", -1)),
        ("Peak MPS allocated", lambda r: r["metrics"].get("mps_memory_peak", {}).get("peak_allocated_gb", -1)),
        ("Post-GC MPS memory", lambda r: r["metrics"].get("mps_memory_post_gc", {}).get("driver_gb", -1)),
        ("Idle MPS memory", lambda r: r["metrics"].get("mps_memory_after_load", {}).get("driver_gb", -1)),
    ]

    for name, extractor in mem_metrics:
        val_b = extractor(before)
        val_a = extractor(after)
        if val_b > 0 and val_a > 0:
            pct = ((val_b - val_a) / val_b) * 100
            change = f"-{pct:.0f}% ✅" if pct > 0 else f"+{abs(pct):.0f}% ⚠️"
            rows.append((name, f"{val_b} GB", f"{val_a} GB", change))
        else:
            rows.append((name, str(val_b), str(val_a), "N/A"))

    # Print table
    col_widths = [
        max(len(r[0]) for r in rows) + 2,
        max(len(str(r[1])) for r in rows) + 2,
        max(len(str(r[2])) for r in rows) + 2,
        max(len(str(r[3])) for r in rows) + 2,
    ]

    header = f"| {'Metric':<{col_widths[0]}} | {label_before:<{col_widths[1]}} | {label_after:<{col_widths[2]}} | {'Change':<{col_widths[3]}} |"
    separator = f"|{'-'*(col_widths[0]+2)}|{'-'*(col_widths[1]+2)}|{'-'*(col_widths[2]+2)}|{'-'*(col_widths[3]+2)}|"

    table_lines = [header, separator]
    for name, val_b, val_a, change in rows:
        line = f"| {name:<{col_widths[0]}} | {val_b:<{col_widths[1]}} | {val_a:<{col_widths[2]}} | {change:<{col_widths[3]}} |"
        table_lines.append(line)

    table = "\n".join(table_lines)
    print(table)

    # Save as markdown
    md_content = f"""# Benchmark Comparison: {label_before} vs {label_after}

**System:** {after['system'].get('chip', 'Apple Silicon')} — {after['system'].get('total_memory_gb', '?')} GB RAM
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Results

{table}

## Environment Details

| | {label_before} | {label_after} |
|---|---|---|
| Python | {py_before} | {py_after} |
| PyTorch | {pt_before} | {pt_after} |
| MPS available | {before['torch']['mps_available']} | {after['torch']['mps_available']} |
| Timestamp | {before['system']['timestamp'][:19]} | {after['system']['timestamp'][:19]} |
"""

    comparison_path = RESULTS_DIR / "comparison.md"
    with open(comparison_path, "w") as f:
        f.write(md_content)

    print(f"\n✅ Comparison saved to: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Stable Diffusion WebUI macOS performance"
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Label for this benchmark run (e.g., 'before', 'after')",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two benchmark results (e.g., --compare before after)",
    )
    parser.add_argument(
        "--system-only",
        action="store_true",
        help="Only collect system info (no model loading or generation)",
    )

    args = parser.parse_args()

    # Clear sys.argv so A1111's cmd_args doesn't crash on --label
    saved_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]

    if args.compare:
        sys.argv = saved_argv
        compare_results(args.compare[0], args.compare[1])
    elif args.label:
        if args.system_only:
            results = {
                "label": args.label,
                "system": get_system_info(),
                "metrics": {
                    "system_memory_idle": get_system_memory(),
                },
            }
            try:
                results["torch"] = get_torch_info()
            except Exception:
                results["torch"] = {"error": "torch not importable"}
            sys.argv = saved_argv
            save_results(results, args.label)
        else:
            run_benchmark(args.label)
            sys.argv = saved_argv
    else:
        sys.argv = saved_argv
        parser.print_help()


if __name__ == "__main__":
    main()
