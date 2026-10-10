"""Kaggle kernel that runs one script from this repo at an exact commit.

scripts/kaggle_run.sh fills in the four placeholders below and pushes this
file as a private Kaggle kernel. On Kaggle it:
  1. checks the GPU is a T4 (stops otherwise)
  2. clones the repo at the given branch and checks out the exact commit SHA
  3. installs requirements.txt (on top of Kaggle's own torch)
  4. links each attached dataset to /tmp/in/<name> (Kaggle's mount layout
     changes over time, so scripts get one fixed path)
  5. runs the script with COQUI_TOS_AGREED=1 and OUT_DIR=/kaggle/working/out
  6. writes out/run_info.json (SHA, script, args, GPU, exit code) and out/log.txt

Not meant to be run by hand.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BRANCH = "__BRANCH__"
SHA = "__SHA__"
SCRIPT = "__SCRIPT__"
ARGS = "__ARGS__"
DATASETS = "__DATASETS__"
KERNELS = "__KERNELS__"
GPU = "__GPU__"  # T4 normally; P100 allowed for training/accuracy runs only

REPO_URL = "https://github.com/arukh281/desi-tts.git"
REPO = "/kaggle/working/desi-tts"
OUT = "/kaggle/working/out"

os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/log.txt", "w")
info = {"branch": BRANCH, "sha": SHA, "script": SCRIPT, "args": ARGS}


def run(cmd: str, cwd: str | None = None, env: dict | None = None) -> int:
    """Run a shell command, streaming its output to the kernel log and out/log.txt."""
    print(f"$ {cmd}", flush=True)
    log.write(f"$ {cmd}\n")
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    log.flush()
    return proc.wait()


def finish(code: int, reason: str = "") -> None:
    shutil.rmtree(REPO, ignore_errors=True)  # don't download the repo back with the outputs
    info["exit_code"] = code
    info["reason"] = reason
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{OUT}/run_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nrun_info: {json.dumps(info)}")
    log.close()
    sys.exit(code)


info["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                     shell=True, capture_output=True, text=True).stdout.strip()
info["gpu"] = gpu
print(f"GPU: {gpu or 'none'}")
info["gpu_requested"] = GPU
if GPU not in gpu:
    finish(2, f"expected a {GPU}, got: {gpu or 'no GPU'}")

if run(f"git clone -q --branch {BRANCH} {REPO_URL} {REPO}") != 0:
    finish(3, "git clone failed")
if run(f"git checkout -q {SHA}", cwd=REPO) != 0:
    finish(3, f"commit {SHA} not found on {BRANCH}")
info["sha_checked_out"] = subprocess.run("git rev-parse HEAD", shell=True, cwd=REPO,
                                         capture_output=True, text=True).stdout.strip()

if run("pip install -q -r requirements.txt", cwd=REPO) != 0:
    finish(4, "pip install failed")
run("pip list 2>/dev/null | grep -i -E '^(torch|torchaudio|torchcodec|coqui-tts|transformers) '")

os.makedirs("/tmp/in", exist_ok=True)
for name in DATASETS.split() + [k.split("/")[-1] for k in KERNELS.split()]:
    found = sorted((p for p in Path("/kaggle/input").rglob(name) if p.is_dir()), key=lambda p: len(p.parts))
    if not found:
        run("find /kaggle/input -maxdepth 4")
        finish(5, f"dataset {name} not found under /kaggle/input")
    os.symlink(found[0], f"/tmp/in/{name}")
    print(f"dataset {name}: {found[0]} -> /tmp/in/{name}")
info["datasets"] = DATASETS
info["kernel_inputs"] = KERNELS

env = dict(os.environ, COQUI_TOS_AGREED="1", OUT_DIR=OUT)
code = run(f"python {SCRIPT} {ARGS}", cwd=REPO, env=env)
finish(code, "" if code == 0 else "script failed")
