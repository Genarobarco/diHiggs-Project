import h5py
import matplotlib.pyplot as plt
import numpy as np

# --- Ajustar ---
FILE_SIG = "logs/<run>/ckpts/<ckpt>__test_sig.h5"
FILE_BKG = "logs/<run>/ckpts/<ckpt>__test_bkg.h5"
OBJ = "events"  # objeto global (data.global_object)
SCORE = "MiModelo_psignal"  # <name>_p<clase de señal>
WPS = [0.5, 0.7, 0.85]  # working points a marcar (eficiencia de señal)


def load(fname):
    with h5py.File(fname) as h:
        return h[OBJ].fields(SCORE)[...].astype(np.float64)


s_sig, s_bkg = load(FILE_SIG), load(FILE_BKG)

# ROC: ordenar todos los scores de mayor a menor y acumular
scores = np.concatenate([s_sig, s_bkg])
is_sig = np.concatenate([np.ones_like(s_sig), np.zeros_like(s_bkg)])
order = np.argsort(-scores, kind="stable")
eff_sig = np.concatenate([[0], np.cumsum(is_sig[order]) / len(s_sig)])
eff_bkg = np.concatenate([[0], np.cumsum(1 - is_sig[order]) / len(s_bkg)])
auc = np.trapezoid(1 - eff_bkg, eff_sig)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(eff_sig, 1 - eff_bkg, label=f"Salt (AUC = {auc:.4f})")
ax.plot([0, 1], [1, 0], "k--", lw=0.8, label="Aleatorio")

for wp in WPS:
    cut = np.percentile(s_sig, 100 * (1 - wp))
    eb = np.mean(s_bkg > cut)
    ax.plot(wp, 1 - eb, "o")
    ax.annotate(f"WP {wp:.0%}: corte={cut:.3f}, $1-\\epsilon_{{bg}}$={1 - eb:.3f}",
                (wp, 1 - eb), textcoords="offset points", xytext=(-10, -15),
                ha="right", fontsize=8)

ax.set_xlabel(r"$\epsilon_{sig}$")
ax.set_ylabel(r"$1 - \epsilon_{bg}$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig("roc.png", dpi=200)
