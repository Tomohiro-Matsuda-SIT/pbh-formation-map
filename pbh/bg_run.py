r"""
Launch bg_node.py on every background node of bg_prior.npz (in parallel, one process per node, BLAS single-threaded).
usage: python bg_run.py [BAND=planck] [NSAMP=100000]
       python bg_run.py validate [NSAMP]   -> one node at E-model alpha=1, w=0, T_reh = Starobinsky value
Node files: bgscan/<band>/<model>_<k>.{json,npz};  logs: logs/bgscan_<band>.log
"""
import os, sys, subprocess, numpy as np, time
from concurrent.futures import ThreadPoolExecutor
import background as bgm
import curvaton_pipeline as cp
from paths import PBH_ROOT, NPROC

band = sys.argv[1] if len(sys.argv) > 1 else "planck"
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
D = os.path.join(PBH_ROOT, "bgscan", band); os.makedirs(D, exist_ok=True)
os.makedirs(os.path.join(PBH_ROOT, "logs"), exist_ok=True)

jobs = []
if band == "validate":
    b = bgm.solve("E", 1.0, 0.0, cp.BG["Treh"], full=True)
    p = os.path.join(D, "E_staro.json"); bgm.write_bg(b, p); jobs.append(p)
else:
    pr = np.load(os.environ.get("PBH_BGPRIOR", os.path.join(PBH_ROOT, "bg_prior.npz")))   # PBH_BGPRIOR: e.g. bg_prior_act.npz
    for model in ("E", "T"):
        key = f"nodes_{model}_{band}"
        if key not in pr:
            print(f"{band}/{model}: no allowed background region -> skipped"); continue
        for k, (la, w, lT) in enumerate(pr[key]):
            p = os.path.join(D, f"{model}_{k:03d}.json")
            if not os.path.exists(p.replace(".json", ".npz")):
                bgm.write_bg(bgm.solve(model, 10**la, w, 10**lT, full=True), p)
                jobs.append(p)

def run(p):
    env = dict(os.environ, PBH_BG=p, PYTHONUTF8="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "bg_node.py", p.replace(".json", ".npz"), str(NS)], env=env,
                       capture_output=True, text=True, cwd=PBH_ROOT)
    msg = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else f"FAILED {p}\n{r.stderr[-2000:]}"
    with open(os.path.join(PBH_ROOT, "logs", f"bgscan_{band}.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

t = time.time()
print(f"{len(jobs)} nodes, NSAMP={NS}, {NPROC} parallel", flush=True)
with ThreadPoolExecutor(NPROC) as ex:
    list(ex.map(run, jobs))
print(f"all done in {(time.time() - t) / 60:.1f} min")
