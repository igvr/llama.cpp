#!/usr/bin/env python3
"""Run a command and retain its output plus sampled process RAM/VRAM."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("a command is required")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    samples = []
    with a.output.with_suffix(".stdout").open("w") as stdout, a.output.with_suffix(".stderr").open("w") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        while child.poll() is None:
            sample = dict(seconds=time.monotonic()-start, gpu_mib=None)
            try:
                for line in Path(f"/proc/{child.pid}/status").read_text().splitlines():
                    if line.split(":")[0] in ("VmRSS", "VmHWM", "VmSwap"):
                        key, value = line.split(":")
                        sample[key+"_bytes"] = int(value.strip().split()[0])*1024
            except FileNotFoundError:
                pass
            try:
                query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                                       text=True, capture_output=True, timeout=5, check=True)
                for line in query.stdout.splitlines():
                    pid, memory = map(str.strip, line.split(","))
                    if pid == str(child.pid):
                        sample["gpu_mib"] = int(memory)
            except (subprocess.SubprocessError, ValueError):
                pass
            samples.append(sample)
            time.sleep(1)
        code = child.wait()
    result = dict(command=command, returncode=code, seconds=time.monotonic()-start,
                  child_maxrss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
                  max_sampled_rss_bytes=max((s.get("VmRSS_bytes", 0) for s in samples), default=0),
                  max_sampled_gpu_mib=max((s["gpu_mib"] for s in samples if s["gpu_mib"] is not None), default=None),
                  samples=samples)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("command", "samples")}, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
