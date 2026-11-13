"""ROC 1 - eff_bg vs eff_sig a partir de las salidas de `salt test`."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# --- Ajustar ---
RUN_DIR = Path("/home/gbarco/DiHiggsProject/OutTrainings/GN2Tracks_v01_20261105-T131007")
EPOCH = "049"
OBJ = "jets"  # global_object del entrenamiento
SIG_CLASS = "hadtaujets"  # el score se llama <modelo>_p<clase>
WPS = [0.6, 0.7, 0.8, 0.9]  # working points a marcar (eficiencia de señal)
OUT_PLOT = "roc_1-effbg.png"

trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0


def find_test_file(sample):
    matches = sorted((RUN_DIR / "ckpts").glob(f"epoch={EPOCH}-*__test_{sample}*.h5"))
    if not matches:
        raise FileNotFoundError(f"No hay test '{sample}' para epoch={EPOCH} en {RUN_DIR / 'ckpts'}")
    return matches[0]


def load(fname):
    with h5py.File(fname, "r") as h:
        d = h[OBJ]
        score = next((n for n in d.dtype.names if n.endswith(f"_p{SIG_CLASS}")), None)
        if score is None:
            raise KeyError(f"No hay '*_p{SIG_CLASS}' en {fname.name}. Campos: {d.dtype.names}")
        return d.fields(score)[:].astype(np.float64)


s_sig = load(find_test_file("signal"))
s_bkg = load(find_test_file("background"))

# ROC: ordenar todos los scores de mayor a menor y acumular
scores = np.concatenate([s_sig, s_bkg])
is_sig = np.concatenate([np.ones_like(s_sig), np.zeros_like(s_bkg)])
order = np.argsort(-scores, kind="stable")
eff_sig = np.concatenate([[0], np.cumsum(is_sig[order]) / len(s_sig)])
eff_bkg = np.concatenate([[0], np.cumsum(1 - is_sig[order]) / len(s_bkg)])
auc = trapezoid(1 - eff_bkg, eff_sig)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(eff_sig, 1 - eff_bkg, label=f"GN2 tracks (AUC = {auc:.4f})")
ax.plot([0, 1], [1, 0], "k--", lw=0.8, label="Aleatorio")

print(f"N señal = {len(s_sig):,}  N fondo = {len(s_bkg):,}  AUC = {auc:.4f}")
for wp in WPS:
    cut = np.percentile(s_sig, 100 * (1 - wp))
    eb = np.mean(s_bkg > cut)
    print(f"  eff_sig = {wp:.0%} -> 1-eff_bg = {1 - eb:.4f}  (corte p_sig > {cut:.4f})")
    ax.plot(wp, 1 - eb, "o")
    ax.annotate(f"{wp:.0%}: $1-\\epsilon_{{bg}}$={1 - eb:.3f}", (wp, 1 - eb),
                textcoords="offset points", xytext=(-8, -14), ha="right", fontsize=8)

ax.set_xlabel(r"$\epsilon_{sig}$")
ax.set_ylabel(r"$1 - \epsilon_{bg}$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=200)
print(f"Guardado: {OUT_PLOT}")


def Asimov(signal, bg):
    return np.sqrt(2*(bg*np.log(bg/(signal+bg)) + signal)) # CORREGIR! esta no es la que va

working_points = np.linspace(0, 1, 100)
significancias = []

for wp in working_points:
    cut = np.percentile(s_sig, np.clip(100 * (1 - wp), 0, 100))
    n_s = int(np.sum(s_sig > cut))
    n_b = int(np.sum(s_bkg > cut))
    significancias.append(Asimov(n_s, n_b) if n_b > 0 else np.nan)

significancias = np.array(significancias)
i = np.nanargmax(significancias)
wp_max, sig_max = working_points[i], significancias[i]

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(working_points, significancias, lw=1.8, label="Asimov")
ax.axvline(wp_max, color="red", ls="--", lw=1.2)
ax.plot(wp_max, sig_max, "r*", ms=15, zorder=5,
        label=rf"máximo: WP = {wp_max:.1%}, $Z$ = {sig_max:.2f}")
ax.annotate(rf"WP = {wp_max:.1%}" "\n" rf"$Z_{{max}}$ = {sig_max:.2f}",
            xy=(wp_max, sig_max), xytext=(12, -28),
            textcoords="offset points", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

ax.set_xlabel(r"Working point ($\epsilon_{sig}$)")
ax.set_ylabel(r"Significance")
ax.grid(alpha=0.3)
ax.legend(loc="lower center")

fig.tight_layout()
fig.savefig(f"Significancia_vs_WP.png", dpi=300)
plt.close()