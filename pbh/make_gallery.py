"""Multi-page PDF gallery of all per-(model, condition) maps in results/ (one page per model)."""
from paths import PBH_ROOT, PBHBOUNDS, CEFFYL, NG15, NPROC
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT = PBH_ROOT + "/results"
models = sorted(d for d in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{d}"))
with PdfPages("results_gallery.pdf") as pdf:
    for m in models:
        pngs = sorted(glob.glob(f"{ROOT}/{m}/*/map.png"))
        if not pngs:
            continue
        for k in range(0, len(pngs), 6):
            chunk = pngs[k:k + 6]
            fig, axs = plt.subplots(3, 2, figsize=(8.27, 11.69))
            for ax in axs.flat:
                ax.axis("off")
            for ax, p in zip(axs.flat, chunk):
                ax.imshow(plt.imread(p))
                d = json.load(open(os.path.join(os.path.dirname(p), "meta.json")))
                ax.set_title(d["description"][:110], fontsize=5.5, wrap=True)
            fig.suptitle(f"{m}  ({k // 6 + 1})", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig, dpi=150); plt.close(fig)
print("pages written")
