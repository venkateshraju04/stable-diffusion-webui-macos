#!/usr/bin/env python3
"""
Benchmark script for Stable Diffusion WebUI macOS.

Captures performance and memory metrics for before/after comparison.
Uses the WebUI's API mode to avoid fragile import chains.

Usage:
    python benchmark.py --label before    # Run benchmark, save as "before"
    python benchmark.py --label after     # Run benchmark, save as "after"
    python benchmark.py --compare before after  # Generate comparison report
    python benchmark.py --system-only     # Only collect system/torch info
"""

import argparse
import gc
import json
import os
import platform
import signal
import subprocess
import sys
import time
import threading
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "benchmark_results"
WEBUI_PORT = 7860
API_URL = f"http://127.0.0.1:{WEBUI_PORT}"


def bytes_to_gb(b):
    """Convert bytes to GB with 2 decimal places."""
    return round(b / (1024 ** 3), 2)


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
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        info["chip"] = chip
    except Exception:
        info["chip"] = "unknown"
    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip())
        info["total_memory_gb"] = round(mem_bytes / (1024 ** 3), 1)
    except Exception:
        info["total_memory_gb"] = "unknown"
    return info


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


def get_torch_info():
    """Collect PyTorch and MPS info (safe standalone import)."""
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        }
    except Exception as e:
        return {"error": str(e)}


class MemoryMonitor:
    """Monitor system memory in a background thread during generation."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []
        self.peak_used_gb = 0
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
        try:
            import psutil
        except ImportError:
            return
        while self._running:
            mem = psutil.virtual_memory()
            used_gb = bytes_to_gb(mem.used)
            self.peak_used_gb = max(self.peak_used_gb, used_gb)
            self.samples.append({
                "time": time.time(),
                "used_gb": used_gb,
                "available_gb": bytes_to_gb(mem.available),
                "percent": mem.percent,
            })
            time.sleep(self.interval)

    def get_stats(self):
        return {
            "peak_system_used_gb": self.peak_used_gb,
            "num_samples": len(self.samples),
            "samples_summary": {
                "min_available_gb": min((s["available_gb"] for s in self.samples), default=0),
                "max_percent": max((s["percent"] for s in self.samples), default=0),
            } if self.samples else {},
        }


def wait_for_webui(timeout=300):
    """Wait for WebUI to become responsive."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{API_URL}/sdapi/v1/progress")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(2)
    return False


def api_request(endpoint, data=None, method="GET"):
    """Make an API request to the WebUI."""
    url = f"{API_URL}{endpoint}"
    if data is not None:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    else:
        req = urllib.request.Request(url, method=method)
    
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_benchmark(label):
    """Run the full benchmark suite."""
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

    # --- System memory at rest ---
    print("[1/5] Capturing idle system memory...")
    results["metrics"]["system_memory_idle"] = get_system_memory()

    # --- Launch WebUI in API mode ---
    print("[2/5] Launching WebUI in API mode...")
    print("       This will load the model and start the server...")

    env = os.environ.copy()
    env["COMMANDLINE_ARGS"] = "--api --nowebui --skip-torch-cuda-test --upcast-sampling"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    launch_start = time.time()
    webui_proc = subprocess.Popen(
        ["bash", "webui.sh"],
        cwd=str(SCRIPT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("       Waiting for server to be ready (this may take a few minutes)...")
    if not wait_for_webui(timeout=300):
        print("       ERROR: WebUI did not start within 5 minutes.")
        webui_proc.terminate()
        results["metrics"]["startup_error"] = "timeout"
        save_results(results, label)
        return results

    startup_time = time.time() - launch_start
    results["metrics"]["startup_time_s"] = round(startup_time, 2)
    print(f"       Server ready! Startup time: {startup_time:.1f}s")

    # Memory after model load
    results["metrics"]["system_memory_after_load"] = get_system_memory()

    # --- Generation benchmark ---
    print("[3/5] Running generation benchmark (512x512, 20 steps, Euler)...")
    monitor = MemoryMonitor(interval=0.3)

    try:
        payload = {
            "prompt": "a photograph of a cat sitting on a windowsill, natural lighting",
            "negative_prompt": "",
            "seed": 42,
            "sampler_name": "Euler",
            "steps": 20,
            "cfg_scale": 7.0,
            "width": 512,
            "height": 512,
            "batch_size": 1,
            "n_iter": 1,
        }

        monitor.start()
        gen_start = time.time()
        gen_result = api_request("/sdapi/v1/txt2img", payload)
        gen_time = time.time() - gen_start
        monitor.stop()

        results["metrics"]["generation_time_s"] = round(gen_time, 2)
        results["metrics"]["generation_its_per_s"] = round(20 / gen_time, 2)
        results["metrics"]["system_memory_peak"] = monitor.get_stats()
        print(f"       Generation time: {gen_time:.2f}s ({20/gen_time:.1f} it/s)")
        print(f"       Peak system memory: {monitor.get_stats()['peak_system_used_gb']} GB")

        # Save sample image
        if gen_result.get("images"):
            import base64
            RESULTS_DIR.mkdir(exist_ok=True)
            img_data = base64.b64decode(gen_result["images"][0])
            img_path = RESULTS_DIR / f"{label}_sample.png"
            with open(img_path, "wb") as f:
                f.write(img_data)
            print(f"       Sample image saved: {img_path.name}")

    except Exception as e:
        monitor.stop()
        results["metrics"]["generation_time_s"] = -1
        results["metrics"]["generation_error"] = str(e)
        print(f"       Generation failed: {e}")

    # --- Post-generation memory ---
    print("[4/5] Post-generation memory snapshot...")
    time.sleep(2)  # Let things settle
    results["metrics"]["system_memory_post_gen"] = get_system_memory()

    # --- Cleanup ---
    print("[5/5] Shutting down WebUI...")
    webui_proc.terminate()
    try:
        webui_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        webui_proc.kill()

    # Final memory after cleanup
    time.sleep(3)
    results["metrics"]["system_memory_after_cleanup"] = get_system_memory()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Python:          {results['system']['python_version'].split()[0]}")
    torch_info = results['torch']
    if 'torch_version' in torch_info:
        print(f"  PyTorch:         {torch_info['torch_version']}")
        print(f"  MPS available:   {torch_info['mps_available']}")
    print(f"  Startup time:    {results['metrics'].get('startup_time_s', 'N/A')}s")
    if results['metrics'].get('generation_time_s', -1) > 0:
        print(f"  Gen time:        {results['metrics']['generation_time_s']}s")
        print(f"  Speed:           {results['metrics']['generation_its_per_s']} it/s")
        print(f"  Peak sys memory: {results['metrics']['system_memory_peak']['peak_system_used_gb']} GB")

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
        print(f"ERROR: {before_path} not found.")
        return
    if not after_path.exists():
        print(f"ERROR: {after_path} not found.")
        return

    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    print(f"\n{'='*70}")
    print(f"  Benchmark Comparison: {label_before} vs {label_after}")
    print(f"{'='*70}\n")

    rows = []

    # Version info
    py_b = before["system"]["python_version"].split()[0]
    py_a = after["system"]["python_version"].split()[0]
    rows.append(("Python version", py_b, py_a,
                  "upgraded ✅" if py_b != py_a else "same"))

    pt_b = before.get("torch", {}).get("torch_version", "?")
    pt_a = after.get("torch", {}).get("torch_version", "?")
    rows.append(("PyTorch version", pt_b, pt_a,
                  "upgraded ✅" if pt_b != pt_a else "same"))

    # Numeric metrics (lower is better)
    for name, key in [
        ("Startup time", "startup_time_s"),
        ("Generation time (512×512)", "generation_time_s"),
    ]:
        vb = before["metrics"].get(key, -1)
        va = after["metrics"].get(key, -1)
        if vb > 0 and va > 0:
            pct = ((vb - va) / vb) * 100
            change = f"-{pct:.0f}% ✅" if pct > 0 else f"+{abs(pct):.0f}% ⚠️"
            rows.append((name, f"{vb}s", f"{va}s", change))
        else:
            rows.append((name, str(vb), str(va), "N/A"))

    # Speed (higher is better)
    vb = before["metrics"].get("generation_its_per_s", -1)
    va = after["metrics"].get("generation_its_per_s", -1)
    if vb > 0 and va > 0:
        pct = ((va - vb) / vb) * 100
        change = f"+{pct:.0f}% ✅" if pct > 0 else f"-{abs(pct):.0f}% ⚠️"
        rows.append(("Generation speed", f"{vb} it/s", f"{va} it/s", change))

    # Memory metrics (lower is better)
    for name, extractor in [
        ("Peak system memory", lambda r: r["metrics"].get("system_memory_peak", {}).get("peak_system_used_gb", -1)),
        ("Memory after load", lambda r: r["metrics"].get("system_memory_after_load", {}).get("used_gb", -1)),
        ("Memory after cleanup", lambda r: r["metrics"].get("system_memory_after_cleanup", {}).get("used_gb", -1)),
    ]:
        vb = extractor(before)
        va = extractor(after)
        if vb > 0 and va > 0:
            pct = ((vb - va) / vb) * 100
            change = f"-{pct:.0f}% ✅" if pct > 0 else f"+{abs(pct):.0f}% ⚠️"
            rows.append((name, f"{vb} GB", f"{va} GB", change))

    # Print table
    col_w = [max(len(r[i]) for r in rows) + 2 for i in range(4)]
    header = f"| {'Metric':<{col_w[0]}} | {label_before:<{col_w[1]}} | {label_after:<{col_w[2]}} | {'Change':<{col_w[3]}} |"
    sep = f"|{'-'*(col_w[0]+2)}|{'-'*(col_w[1]+2)}|{'-'*(col_w[2]+2)}|{'-'*(col_w[3]+2)}|"

    table_lines = [header, sep]
    for row in rows:
        table_lines.append(f"| {row[0]:<{col_w[0]}} | {row[1]:<{col_w[1]}} | {row[2]:<{col_w[2]}} | {row[3]:<{col_w[3]}} |")
    table = "\n".join(table_lines)
    print(table)

    # Save markdown
    md = f"""# Benchmark Comparison: {label_before} vs {label_after}

**System:** {after['system'].get('chip', 'Apple Silicon')} — {after['system'].get('total_memory_gb', '?')} GB RAM
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Results

{table}

## Environment

| | {label_before} | {label_after} |
|---|---|---|
| Python | {py_b} | {py_a} |
| PyTorch | {pt_b} | {pt_a} |
"""
    comp_path = RESULTS_DIR / "comparison.md"
    with open(comp_path, "w") as f:
        f.write(md)
    print(f"\n✅ Comparison saved to: {comp_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Stable Diffusion WebUI macOS performance"
    )
    parser.add_argument("--label", type=str, help="Label for this run (e.g., 'before', 'after')")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="Compare two results")
    parser.add_argument("--system-only", action="store_true", help="Only collect system info")

    args, _ = parser.parse_known_args()  # Use parse_known_args to ignore WebUI args

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.label:
        if args.system_only:
            results = {
                "label": args.label,
                "system": get_system_info(),
                "torch": get_torch_info(),
                "metrics": {"system_memory_idle": get_system_memory()},
            }
            save_results(results, args.label)
        else:
            run_benchmark(args.label)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
