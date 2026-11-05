"""ROC (rechazo de fondo vs. eficiencia de señal) a partir de las salidas de `salt test`."""

from pathlib import Path

import h5py
import numpy as np
from ftag import Flavours

try:
    from ftag.utils import calculate_rejection
except ImportError:
    try:
        from ftag.utils.metrics import calculate_rejection
    except ImportError:
        from puma.metrics import calc_rej as calculate_rejection

from puma import Roc, RocPlot

# ----------------------------------------------------------------------------------------------
# Configuración
BASE_DIR = Path("/home/gbarco/DiHiggsProject/OutTrainings")
RUN_NAME = "GN2Tracks_v01"  # nombre de la corrida (-n de salt fit)

# (time_stamp, época, etiqueta en el gráfico)
data_list = [
    ("20261104-T200117", "047", "GN2Tracks v01"),
]

SIG_EFF = np.linspace(0.3, 1.0, 100)  # eficiencias de señal donde se evalúa el rechazo
WPS = [0.6, 0.7, 0.8, 0.9]  # working points a imprimir
OUT_PLOT = "roc_tracks.png"
EPS = 1e-8


# ----------------------------------------------------------------------------------------------
# Funciones
def find_test_file(run_dir: Path, n_epoch: str, sample: str) -> Path:
    """Busca el .h5 de test sin depender del valor de val_loss en el nombre."""
    matches = sorted((run_dir / "ckpts").glob(f"epoch={n_epoch}-*__test_{sample}*.h5"))
    if not matches:
        raise FileNotFoundError(f"No hay archivo de test '{sample}' para epoch={n_epoch} en {run_dir / 'ckpts'}")
    if len(matches) > 1:
        print(f"[aviso] varios archivos para {sample}, uso {matches[0].name}: {[m.name for m in matches]}")
    return matches[0]


def load_preds(path: Path):
    """Devuelve (p_tau, p_qcd, label) del dataset 'jets'. Detecta solo el prefijo del modelo."""
    with h5py.File(path, "r") as f:
        jets = f["jets"]
        names = jets.dtype.names
        prefix = next((n[: -len("_phadtaujets")] for n in names if n.endswith("_phadtaujets")), None)
        if prefix is None:
            raise KeyError(f"No encuentro '*_phadtaujets' en {path.name}. Campos: {names}")
        p_tau = jets.fields(f"{prefix}_phadtaujets")[:]
        p_qcd = jets.fields(f"{prefix}_pqcdjets")[:]
        label = jets.fields("flavour_label")[:]
    return p_tau, p_qcd, label, prefix


def discriminant(p_tau, p_qcd):
    """D = log(p_tau / p_qcd). Con 2 clases es monótono en p_tau (misma ROC), pero mejor resuelto cerca de 1."""
    return np.log((p_tau + EPS) / (p_qcd + EPS))


def auc(sig_disc, bkg_disc):
    """AUC = P(D_señal > D_fondo), vía rangos (Mann-Whitney)."""
    d = np.concatenate([sig_disc, bkg_disc])
    order = d.argsort()
    ranks = np.empty(len(d))
    ranks[order] = np.arange(1, len(d) + 1)
    # rangos promedio para empates
    _, inv, counts = np.unique(d, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n_s, n_b = len(sig_disc), len(bkg_disc)
    return (ranks[:n_s].sum() - n_s * (n_s + 1) / 2) / (n_s * n_b)


def disc_rej(time_stamp: str, n_epoch: str):
    run_dir = BASE_DIR / f"{RUN_NAME}_{time_stamp}"
    sig_file = find_test_file(run_dir, n_epoch, "signal")
    bkg_file = find_test_file(run_dir, n_epoch, "background")

    pt_s, pq_s, lab_s, prefix = load_preds(sig_file)
    pt_b, pq_b, lab_b, _ = load_preds(bkg_file)

    # chequeo: el archivo de señal debe tener solo label 0 y el de fondo solo label 1
    if not (np.all(lab_s == 0) and np.all(lab_b == 1)):
        print(f"[aviso] labels inesperados: señal {np.unique(lab_s)}, fondo {np.unique(lab_b)}")

    d_sig = discriminant(pt_s, pq_s)
    d_bkg = discriminant(pt_b, pq_b)

    rej = calculate_rejection(d_sig, d_bkg, SIG_EFF)

    print(f"\n{RUN_NAME}_{time_stamp}  epoch={n_epoch}  (salida: {prefix})")
    print(f"  N señal = {len(d_sig):,}   N fondo = {len(d_bkg):,}   AUC = {auc(d_sig, d_bkg):.4f}")
    rej_wp, cuts = calculate_rejection(d_sig, d_bkg, WPS, return_cuts=True)
    for wp, r, c in zip(WPS, rej_wp, cuts):
        print(f"  eff_sig = {wp:.0%}  ->  rechazo = {r:10.2f}   (corte D > {c:.3f})")

    return SIG_EFF, rej, len(d_bkg)


# ----------------------------------------------------------------------------------------------
# Plot
plot_roc = RocPlot(
    n_ratio_panels=1,
    ylabel="Background rejection",
    xlabel="$\\tau$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13.6$ TeV \n $\\gamma* \\to \\tau \\tau$ signal",
    figsize=(6.5, 6),
    y_scale=1.4,
)

for i, (time_stamp, n_epoch, label) in enumerate(data_list):
    sig_eff, rej, n_bkg = disc_rej(time_stamp, n_epoch)
    plot_roc.add_roc(
        Roc(
            sig_eff,
            rej,
            n_test=n_bkg,
            rej_class=Flavours["qcd"],
            signal_class="taujets",
            label=label,
        ),
        reference=(i == 0),
    )

plot_roc.set_ratio_class(1, Flavours["qcd"])
plot_roc.draw()
plot_roc.savefig(OUT_PLOT, transparent=False)
print(f"\nGuardado: {OUT_PLOT}")
