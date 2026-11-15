"""Build a kernel that runs two train.py experiments in parallel, one per T4. Usage: build_pair.py NAME 'envA' 'envB' (env as k=v,k=v)."""
import json, os, sys

name, a, b = sys.argv[1:4]
src = open("train.py").read()
exps = [dict(kv.split("=") for kv in x.split(",")) for x in (a, b)]
code = f'''import os, subprocess, sys
SRC = {src!r}
open("/kaggle/working/train_core.py", "w").write(SRC)
EXPS = {exps!r}
if any("ARCH" in e for e in EXPS):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True)
procs = []
for gpu, env in enumerate(EXPS):
    tag = env.pop("TAG")
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT=f"/kaggle/working/{{tag}}", BUDGET="9000", **env)
    procs.append((tag, subprocess.Popen([sys.executable, "/kaggle/working/train_core.py"], env=e,
                  stdout=open(f"/kaggle/working/{{tag}}.log", "w"), stderr=subprocess.STDOUT)))
for tag, p in procs:
    p.wait()
    print("=====", tag, "exit", p.returncode, flush=True)
    print(open(f"/kaggle/working/{{tag}}.log").read()[-3000:], flush=True)
'''
k = f"kern/{name}"
os.makedirs(k, exist_ok=True)
open(f"{k}/train.py", "w").write(code)
json.dump({"id": f"mateobuiles/filament-{name}", "title": f"filament-{name}", "code_file": "train.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": any("ARCH" in e for e in exps),
           "dataset_sources": ["mateobuiles/resnet34-imagenet"], "competition_sources": ["filament-segmentation-2026"]},
          open(f"{k}/kernel-metadata.json", "w"), indent=2)
