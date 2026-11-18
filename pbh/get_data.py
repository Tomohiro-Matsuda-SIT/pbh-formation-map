r"""
Unpack the stored samples downloaded from Zenodo into this repository.
usage (from the pbh/ directory):  python get_data.py PATH/TO/pbh-data-results.zip [more zip files ...]

The Zenodo record (DOI given in README.md) contains three archives; each unpacks into its place below the repository
root and checks the MD5 sums listed in manifest.json if that file is given or lies next to the archives:
    pbh-data-results.zip   pbh/results/<mechanism>/<condition>/samples.npz   (all stored treatments)
    pbh-data-bgscan.zip    pbh/bgscan/...                                    (inflation-model family, variations)
    pbh-data-samples.zip   pbh/fopt_dw_samples.npz, pbh/axion_prior_samples_1e6.npz
With these files the figures and numbers can be rebuilt without rerunning the mechanisms (e.g. fig_combined.py,
measure_checks.py, measure_plots.py, robustness.py analyze).
"""
import os, sys, json, zipfile, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(2**24), b""):
            h.update(b)
    return h.hexdigest()

for zp in sys.argv[1:]:
    man = os.path.join(os.path.dirname(os.path.abspath(zp)), "manifest.json")
    if os.path.exists(man):
        ref = json.load(open(man)).get(os.path.basename(zp), {}).get("md5")
        if ref and md5(zp) != ref:
            sys.exit(f"MD5 mismatch for {zp}; download it again")
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            dest = os.path.normpath(os.path.join(ROOT, n))
            if not dest.startswith(ROOT + os.sep) or not n.startswith("pbh/"):
                sys.exit(f"unexpected path in archive: {n}")
        z.extractall(ROOT)
        print(f"{os.path.basename(zp)}: {len(z.namelist())} files unpacked")
